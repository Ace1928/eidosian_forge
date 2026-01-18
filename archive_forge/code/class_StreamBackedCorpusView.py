import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
class StreamBackedCorpusView(AbstractLazySequence):
    """
    A 'view' of a corpus file, which acts like a sequence of tokens:
    it can be accessed by index, iterated over, etc.  However, the
    tokens are only constructed as-needed -- the entire corpus is
    never stored in memory at once.

    The constructor to ``StreamBackedCorpusView`` takes two arguments:
    a corpus fileid (specified as a string or as a ``PathPointer``);
    and a block reader.  A "block reader" is a function that reads
    zero or more tokens from a stream, and returns them as a list.  A
    very simple example of a block reader is:

        >>> def simple_block_reader(stream):
        ...     return stream.readline().split()

    This simple block reader reads a single line at a time, and
    returns a single token (consisting of a string) for each
    whitespace-separated substring on the line.

    When deciding how to define the block reader for a given
    corpus, careful consideration should be given to the size of
    blocks handled by the block reader.  Smaller block sizes will
    increase the memory requirements of the corpus view's internal
    data structures (by 2 integers per block).  On the other hand,
    larger block sizes may decrease performance for random access to
    the corpus.  (But note that larger block sizes will *not*
    decrease performance for iteration.)

    Internally, ``CorpusView`` maintains a partial mapping from token
    index to file position, with one entry per block.  When a token
    with a given index *i* is requested, the ``CorpusView`` constructs
    it as follows:

      1. First, it searches the toknum/filepos mapping for the token
         index closest to (but less than or equal to) *i*.

      2. Then, starting at the file position corresponding to that
         index, it reads one block at a time using the block reader
         until it reaches the requested token.

    The toknum/filepos mapping is created lazily: it is initially
    empty, but every time a new block is read, the block's
    initial token is added to the mapping.  (Thus, the toknum/filepos
    map has one entry per block.)

    In order to increase efficiency for random access patterns that
    have high degrees of locality, the corpus view may cache one or
    more blocks.

    :note: Each ``CorpusView`` object internally maintains an open file
        object for its underlying corpus file.  This file should be
        automatically closed when the ``CorpusView`` is garbage collected,
        but if you wish to close it manually, use the ``close()``
        method.  If you access a ``CorpusView``'s items after it has been
        closed, the file object will be automatically re-opened.

    :warning: If the contents of the file are modified during the
        lifetime of the ``CorpusView``, then the ``CorpusView``'s behavior
        is undefined.

    :warning: If a unicode encoding is specified when constructing a
        ``CorpusView``, then the block reader may only call
        ``stream.seek()`` with offsets that have been returned by
        ``stream.tell()``; in particular, calling ``stream.seek()`` with
        relative offsets, or with offsets based on string lengths, may
        lead to incorrect behavior.

    :ivar _block_reader: The function used to read
        a single block from the underlying file stream.
    :ivar _toknum: A list containing the token index of each block
        that has been processed.  In particular, ``_toknum[i]`` is the
        token index of the first token in block ``i``.  Together
        with ``_filepos``, this forms a partial mapping between token
        indices and file positions.
    :ivar _filepos: A list containing the file position of each block
        that has been processed.  In particular, ``_toknum[i]`` is the
        file position of the first character in block ``i``.  Together
        with ``_toknum``, this forms a partial mapping between token
        indices and file positions.
    :ivar _stream: The stream used to access the underlying corpus file.
    :ivar _len: The total number of tokens in the corpus, if known;
        or None, if the number of tokens is not yet known.
    :ivar _eofpos: The character position of the last character in the
        file.  This is calculated when the corpus view is initialized,
        and is used to decide when the end of file has been reached.
    :ivar _cache: A cache of the most recently read block.  It
       is encoded as a tuple (start_toknum, end_toknum, tokens), where
       start_toknum is the token index of the first token in the block;
       end_toknum is the token index of the first token not in the
       block; and tokens is a list of the tokens in the block.
    """

    def __init__(self, fileid, block_reader=None, startpos=0, encoding='utf8'):
        """
        Create a new corpus view, based on the file ``fileid``, and
        read with ``block_reader``.  See the class documentation
        for more information.

        :param fileid: The path to the file that is read by this
            corpus view.  ``fileid`` can either be a string or a
            ``PathPointer``.

        :param startpos: The file position at which the view will
            start reading.  This can be used to skip over preface
            sections.

        :param encoding: The unicode encoding that should be used to
            read the file's contents.  If no encoding is specified,
            then the file's contents will be read as a non-unicode
            string (i.e., a str).
        """
        if block_reader:
            self.read_block = block_reader
        self._toknum = [0]
        self._filepos = [startpos]
        self._encoding = encoding
        self._len = None
        self._fileid = fileid
        self._stream = None
        self._current_toknum = None
        'This variable is set to the index of the next token that\n           will be read, immediately before ``self.read_block()`` is\n           called.  This is provided for the benefit of the block\n           reader, which under rare circumstances may need to know\n           the current token number.'
        self._current_blocknum = None
        'This variable is set to the index of the next block that\n           will be read, immediately before ``self.read_block()`` is\n           called.  This is provided for the benefit of the block\n           reader, which under rare circumstances may need to know\n           the current block number.'
        try:
            if isinstance(self._fileid, PathPointer):
                self._eofpos = self._fileid.file_size()
            else:
                self._eofpos = os.stat(self._fileid).st_size
        except Exception as exc:
            raise ValueError(f'Unable to open or access {fileid!r} -- {exc}') from exc
        self._cache = (-1, -1, None)
    fileid = property(lambda self: self._fileid, doc='\n        The fileid of the file that is accessed by this view.\n\n        :type: str or PathPointer')

    def read_block(self, stream):
        """
        Read a block from the input stream.

        :return: a block of tokens from the input stream
        :rtype: list(any)
        :param stream: an input stream
        :type stream: stream
        """
        raise NotImplementedError('Abstract Method')

    def _open(self):
        """
        Open the file stream associated with this corpus view.  This
        will be called performed if any value is read from the view
        while its file stream is closed.
        """
        if isinstance(self._fileid, PathPointer):
            self._stream = self._fileid.open(self._encoding)
        elif self._encoding:
            self._stream = SeekableUnicodeStreamReader(open(self._fileid, 'rb'), self._encoding)
        else:
            self._stream = open(self._fileid, 'rb')

    def close(self):
        """
        Close the file stream associated with this corpus view.  This
        can be useful if you are worried about running out of file
        handles (although the stream should automatically be closed
        upon garbage collection of the corpus view).  If the corpus
        view is accessed after it is closed, it will be automatically
        re-opened.
        """
        if self._stream is not None:
            self._stream.close()
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __len__(self):
        if self._len is None:
            for tok in self.iterate_from(self._toknum[-1]):
                pass
        return self._len

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop = slice_bounds(self, i)
            offset = self._cache[0]
            if offset <= start and stop <= self._cache[1]:
                return self._cache[2][start - offset:stop - offset]
            return LazySubsequence(self, start, stop)
        else:
            if i < 0:
                i += len(self)
            if i < 0:
                raise IndexError('index out of range')
            offset = self._cache[0]
            if offset <= i < self._cache[1]:
                return self._cache[2][i - offset]
            try:
                return next(self.iterate_from(i))
            except StopIteration as e:
                raise IndexError('index out of range') from e

    def iterate_from(self, start_tok):
        if self._cache[0] <= start_tok < self._cache[1]:
            for tok in self._cache[2][start_tok - self._cache[0]:]:
                yield tok
                start_tok += 1
        if start_tok < self._toknum[-1]:
            block_index = bisect.bisect_right(self._toknum, start_tok) - 1
            toknum = self._toknum[block_index]
            filepos = self._filepos[block_index]
        else:
            block_index = len(self._toknum) - 1
            toknum = self._toknum[-1]
            filepos = self._filepos[-1]
        if self._stream is None:
            self._open()
        if self._eofpos == 0:
            self._len = 0
        while filepos < self._eofpos:
            self._stream.seek(filepos)
            self._current_toknum = toknum
            self._current_blocknum = block_index
            tokens = self.read_block(self._stream)
            assert isinstance(tokens, (tuple, list, AbstractLazySequence)), 'block reader %s() should return list or tuple.' % self.read_block.__name__
            num_toks = len(tokens)
            new_filepos = self._stream.tell()
            assert new_filepos > filepos, 'block reader %s() should consume at least 1 byte (filepos=%d)' % (self.read_block.__name__, filepos)
            self._cache = (toknum, toknum + num_toks, list(tokens))
            assert toknum <= self._toknum[-1]
            if num_toks > 0:
                block_index += 1
                if toknum == self._toknum[-1]:
                    assert new_filepos > self._filepos[-1]
                    self._filepos.append(new_filepos)
                    self._toknum.append(toknum + num_toks)
                else:
                    assert new_filepos == self._filepos[block_index], 'inconsistent block reader (num chars read)'
                    assert toknum + num_toks == self._toknum[block_index], 'inconsistent block reader (num tokens returned)'
            if new_filepos == self._eofpos:
                self._len = toknum + num_toks
            for tok in tokens[max(0, start_tok - toknum):]:
                yield tok
            assert new_filepos <= self._eofpos
            if new_filepos == self._eofpos:
                break
            toknum += num_toks
            filepos = new_filepos
        assert self._len is not None
        self.close()

    def __add__(self, other):
        return concat([self, other])

    def __radd__(self, other):
        return concat([other, self])

    def __mul__(self, count):
        return concat([self] * count)

    def __rmul__(self, count):
        return concat([self] * count)
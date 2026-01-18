import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
class NKJPCorpusReader(XMLCorpusReader):
    WORDS_MODE = 0
    SENTS_MODE = 1
    HEADER_MODE = 2
    RAW_MODE = 3

    def __init__(self, root, fileids='.*'):
        """
        Corpus reader designed to work with National Corpus of Polish.
        See http://nkjp.pl/ for more details about NKJP.
        use example:
        import nltk
        import nkjp
        from nkjp import NKJPCorpusReader
        x = NKJPCorpusReader(root='/home/USER/nltk_data/corpora/nkjp/', fileids='') # obtain the whole corpus
        x.header()
        x.raw()
        x.words()
        x.tagged_words(tags=['subst', 'comp'])  #Link to find more tags: nkjp.pl/poliqarp/help/ense2.html
        x.sents()
        x = NKJPCorpusReader(root='/home/USER/nltk_data/corpora/nkjp/', fileids='Wilk*') # obtain particular file(s)
        x.header(fileids=['WilkDom', '/home/USER/nltk_data/corpora/nkjp/WilkWilczy'])
        x.tagged_words(fileids=['WilkDom', '/home/USER/nltk_data/corpora/nkjp/WilkWilczy'], tags=['subst', 'comp'])
        """
        if isinstance(fileids, str):
            XMLCorpusReader.__init__(self, root, fileids + '.*/header.xml')
        else:
            XMLCorpusReader.__init__(self, root, [fileid + '/header.xml' for fileid in fileids])
        self._paths = self.get_paths()

    def get_paths(self):
        return [os.path.join(str(self._root), f.split('header.xml')[0]) for f in self._fileids]

    def fileids(self):
        """
        Returns a list of file identifiers for the fileids that make up
        this corpus.
        """
        return [f.split('header.xml')[0] for f in self._fileids]

    def _view(self, filename, tags=None, **kwargs):
        """
        Returns a view specialised for use with particular corpus file.
        """
        mode = kwargs.pop('mode', NKJPCorpusReader.WORDS_MODE)
        if mode is NKJPCorpusReader.WORDS_MODE:
            return NKJPCorpus_Morph_View(filename, tags=tags)
        elif mode is NKJPCorpusReader.SENTS_MODE:
            return NKJPCorpus_Segmentation_View(filename, tags=tags)
        elif mode is NKJPCorpusReader.HEADER_MODE:
            return NKJPCorpus_Header_View(filename, tags=tags)
        elif mode is NKJPCorpusReader.RAW_MODE:
            return NKJPCorpus_Text_View(filename, tags=tags, mode=NKJPCorpus_Text_View.RAW_MODE)
        else:
            raise NameError('No such mode!')

    def add_root(self, fileid):
        """
        Add root if necessary to specified fileid.
        """
        if self.root in fileid:
            return fileid
        return self.root + fileid

    @_parse_args
    def header(self, fileids=None, **kwargs):
        """
        Returns header(s) of specified fileids.
        """
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.HEADER_MODE, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def sents(self, fileids=None, **kwargs):
        """
        Returns sentences in specified fileids.
        """
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.SENTS_MODE, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def words(self, fileids=None, **kwargs):
        """
        Returns words in specified fileids.
        """
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.WORDS_MODE, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def tagged_words(self, fileids=None, **kwargs):
        """
        Call with specified tags as a list, e.g. tags=['subst', 'comp'].
        Returns tagged words in specified fileids.
        """
        tags = kwargs.pop('tags', [])
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.WORDS_MODE, tags=tags, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def raw(self, fileids=None, **kwargs):
        """
        Returns words in specified fileids.
        """
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.RAW_MODE, **kwargs).handle_query() for fileid in fileids])
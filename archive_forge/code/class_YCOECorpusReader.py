import os
import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import RegexpTokenizer
class YCOECorpusReader(CorpusReader):
    """
    Corpus reader for the York-Toronto-Helsinki Parsed Corpus of Old
    English Prose (YCOE), a 1.5 million word syntactically-annotated
    corpus of Old English prose texts.
    """

    def __init__(self, root, encoding='utf8'):
        CorpusReader.__init__(self, root, [], encoding)
        self._psd_reader = YCOEParseCorpusReader(self.root.join('psd'), '.*', '.psd', encoding=encoding)
        self._pos_reader = YCOETaggedCorpusReader(self.root.join('pos'), '.*', '.pos')
        documents = {f[:-4] for f in self._psd_reader.fileids()}
        if {f[:-4] for f in self._pos_reader.fileids()} != documents:
            raise ValueError('Items in "psd" and "pos" subdirectories do not match.')
        fileids = sorted(['%s.psd' % doc for doc in documents] + ['%s.pos' % doc for doc in documents])
        CorpusReader.__init__(self, root, fileids, encoding)
        self._documents = sorted(documents)

    def documents(self, fileids=None):
        """
        Return a list of document identifiers for all documents in
        this corpus, or for the documents with the given file(s) if
        specified.
        """
        if fileids is None:
            return self._documents
        if isinstance(fileids, str):
            fileids = [fileids]
        for f in fileids:
            if f not in self._fileids:
                raise KeyError('File id %s not found' % fileids)
        return sorted({f[:-4] for f in fileids})

    def fileids(self, documents=None):
        """
        Return a list of file identifiers for the files that make up
        this corpus, or that store the given document(s) if specified.
        """
        if documents is None:
            return self._fileids
        elif isinstance(documents, str):
            documents = [documents]
        return sorted(set(['%s.pos' % doc for doc in documents] + ['%s.psd' % doc for doc in documents]))

    def _getfileids(self, documents, subcorpus):
        """
        Helper that selects the appropriate fileids for a given set of
        documents from a given subcorpus (pos or psd).
        """
        if documents is None:
            documents = self._documents
        else:
            if isinstance(documents, str):
                documents = [documents]
            for document in documents:
                if document not in self._documents:
                    if document[-4:] in ('.pos', '.psd'):
                        raise ValueError('Expected a document identifier, not a file identifier.  (Use corpus.documents() to get a list of document identifiers.')
                    else:
                        raise ValueError('Document identifier %s not found' % document)
        return [f'{d}.{subcorpus}' for d in documents]

    def words(self, documents=None):
        return self._pos_reader.words(self._getfileids(documents, 'pos'))

    def sents(self, documents=None):
        return self._pos_reader.sents(self._getfileids(documents, 'pos'))

    def paras(self, documents=None):
        return self._pos_reader.paras(self._getfileids(documents, 'pos'))

    def tagged_words(self, documents=None):
        return self._pos_reader.tagged_words(self._getfileids(documents, 'pos'))

    def tagged_sents(self, documents=None):
        return self._pos_reader.tagged_sents(self._getfileids(documents, 'pos'))

    def tagged_paras(self, documents=None):
        return self._pos_reader.tagged_paras(self._getfileids(documents, 'pos'))

    def parsed_sents(self, documents=None):
        return self._psd_reader.parsed_sents(self._getfileids(documents, 'psd'))
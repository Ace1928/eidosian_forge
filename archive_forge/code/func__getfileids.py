import os
import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import RegexpTokenizer
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
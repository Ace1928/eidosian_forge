import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
def __fileids(self, fileids):
    if fileids is None:
        fileids = self._fileids
    elif isinstance(fileids, str):
        fileids = [fileids]
    fileids = filter(lambda x: x in self._fileids, fileids)
    fileids = filter(lambda x: x not in ['oana-bg.xml', 'oana-mk.xml'], fileids)
    if not fileids:
        print('No valid multext-east file specified')
    return fileids
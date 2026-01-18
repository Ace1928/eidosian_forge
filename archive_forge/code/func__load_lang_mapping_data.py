import re
from os import path
from nltk.corpus.reader import CorpusReader
from nltk.data import ZipFilePathPointer
from nltk.probability import FreqDist
def _load_lang_mapping_data(self):
    """Load language mappings between codes and description from table.txt"""
    if isinstance(self.root, ZipFilePathPointer):
        raise RuntimeError("Please install the 'crubadan' corpus first, use nltk.download()")
    mapper_file = path.join(self.root, self._LANG_MAPPER_FILE)
    if self._LANG_MAPPER_FILE not in self.fileids():
        raise RuntimeError('Could not find language mapper file: ' + mapper_file)
    with open(mapper_file, encoding='utf-8') as raw:
        strip_raw = raw.read().strip()
        self._lang_mapping_data = [row.split('\t') for row in strip_raw.split('\n')]
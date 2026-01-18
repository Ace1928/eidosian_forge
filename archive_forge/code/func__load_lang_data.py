import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def _load_lang_data(self, lang):
    """load the wordnet data of the requested language from the file to
        the cache, _lang_data"""
    if lang in self._lang_data:
        return
    if self._omw_reader and (not self.omw_langs):
        self.add_omw()
    if lang not in self.langs():
        raise WordNetError('Language is not supported.')
    if self._exomw_reader and lang not in self.omw_langs:
        reader = self._exomw_reader
    else:
        reader = self._omw_reader
    prov = self.provenances[lang]
    if prov in ['cldr', 'wikt']:
        prov2 = prov
    else:
        prov2 = 'data'
    with reader.open(f'{prov}/wn-{prov2}-{lang.split('_')[0]}.tab') as fp:
        self.custom_lemmas(fp, lang)
    self.disable_custom_lemmas(lang)
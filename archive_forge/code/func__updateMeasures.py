import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def _updateMeasures(self):
    if self._measuresNeedUpdate:
        self._tp = self._guessed & self._correct
        self._fn = self._correct - self._guessed
        self._fp = self._guessed - self._correct
        self._tp_num = len(self._tp)
        self._fp_num = len(self._fp)
        self._fn_num = len(self._fn)
        self._measuresNeedUpdate = False
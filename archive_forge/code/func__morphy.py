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
def _morphy(self, form, pos, check_exceptions=True):
    exceptions = self._exception_map[pos]
    substitutions = self.MORPHOLOGICAL_SUBSTITUTIONS[pos]

    def apply_rules(forms):
        return [form[:-len(old)] + new for form in forms for old, new in substitutions if form.endswith(old)]

    def filter_forms(forms):
        result = []
        seen = set()
        for form in forms:
            if form in self._lemma_pos_offset_map:
                if pos in self._lemma_pos_offset_map[form]:
                    if form not in seen:
                        result.append(form)
                        seen.add(form)
        return result
    if check_exceptions:
        if form in exceptions:
            return filter_forms([form] + exceptions[form])
    forms = apply_rules([form])
    results = filter_forms([form] + forms)
    if results:
        return results
    while forms:
        forms = apply_rules(forms)
        results = filter_forms(forms)
        if results:
            return results
    return []
import warnings
from collections import defaultdict
from math import log
def __build_translation(self, hypothesis, output):
    if hypothesis.previous is None:
        return
    self.__build_translation(hypothesis.previous, output)
    output.extend(hypothesis.trg_phrase)
import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
class BoxerIndexed(AbstractBoxerDrs):

    def __init__(self, discourse_id, sent_index, word_indices):
        AbstractBoxerDrs.__init__(self)
        self.discourse_id = discourse_id
        self.sent_index = sent_index
        self.word_indices = word_indices

    def atoms(self):
        return {self}

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.discourse_id == other.discourse_id and (self.sent_index == other.sent_index) and (self.word_indices == other.word_indices) and reduce(operator.and_, (s == o for s, o in zip(self, other)))

    def __ne__(self, other):
        return not self == other
    __hash__ = AbstractBoxerDrs.__hash__

    def __repr__(self):
        s = '{}({}, {}, [{}]'.format(self._pred(), self.discourse_id, self.sent_index, ', '.join(('%s' % wi for wi in self.word_indices)))
        for v in self:
            s += ', %s' % v
        return s + ')'
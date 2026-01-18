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
class AbstractBoxerDrs:

    def variables(self):
        """
        :return: (set<variables>, set<events>, set<propositions>)
        """
        variables, events, propositions = self._variables()
        return (variables - (events | propositions), events, propositions - events)

    def variable_types(self):
        vartypes = {}
        for t, vars in zip(('z', 'e', 'p'), self.variables()):
            for v in vars:
                vartypes[v] = t
        return vartypes

    def _variables(self):
        """
        :return: (set<variables>, set<events>, set<propositions>)
        """
        return (set(), set(), set())

    def atoms(self):
        return set()

    def clean(self):
        return self

    def _clean_name(self, name):
        return name.replace('-', '_').replace("'", '_')

    def renumber_sentences(self, f):
        return self

    def __hash__(self):
        return hash(f'{self}')
import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
class BindingException(Exception):

    def __init__(self, arg):
        if isinstance(arg, tuple):
            Exception.__init__(self, "'%s' cannot be bound to '%s'" % arg)
        else:
            Exception.__init__(self, arg)
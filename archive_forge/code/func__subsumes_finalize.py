import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _subsumes_finalize(first, second, bindings, used, skipped, debug):
    if not len(skipped[0]) and (not len(first)):
        return [True]
    else:
        return []
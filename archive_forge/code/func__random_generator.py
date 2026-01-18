import random
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect
from itertools import accumulate
from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import Vocabulary
def _random_generator(seed_or_generator):
    if isinstance(seed_or_generator, random.Random):
        return seed_or_generator
    return random.Random(seed_or_generator)
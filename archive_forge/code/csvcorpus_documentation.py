from __future__ import with_statement
import logging
import csv
import itertools
from gensim import interfaces, utils
Iterate over the corpus, returning one BoW vector at a time.

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        
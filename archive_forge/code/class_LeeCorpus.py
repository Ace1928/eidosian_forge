import contextlib
import tempfile
import os
import shutil
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
class LeeCorpus:

    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for line in f:
                yield simple_preprocess(line)
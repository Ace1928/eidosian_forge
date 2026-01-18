import os
from nltk.internals import config_java
import subprocess
from nltk.collocations import *
from nltk.decorators import decorator, memoize
from nltk.featstruct import *
from nltk.grammar import *
from nltk.probability import *
from nltk.text import *
from nltk.util import *
from nltk.jsontags import *
from nltk.chunk import *
from nltk.classify import *
from nltk.inference import *
from nltk.metrics import *
from nltk.parse import *
from nltk.tag import *
from nltk.tokenize import *
from nltk.translate import *
from nltk.tree import *
from nltk.sem import *
from nltk.stem import *
from nltk import lazyimport
from nltk.downloader import download, download_shell
from nltk import ccg, chunk, classify, collocations
from nltk import data, featstruct, grammar, help, inference, metrics
from nltk import misc, parse, probability, sem, stem, wsd
from nltk import tag, tbl, text, tokenize, translate, tree, util
def _fake_Popen(*args, **kwargs):
    raise NotImplementedError('subprocess.Popen is not supported.')
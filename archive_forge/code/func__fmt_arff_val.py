import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
def _fmt_arff_val(self, fval):
    if fval is None:
        return '?'
    elif isinstance(fval, (bool, int)):
        return '%s' % fval
    elif isinstance(fval, float):
        return '%r' % fval
    else:
        return '%r' % fval
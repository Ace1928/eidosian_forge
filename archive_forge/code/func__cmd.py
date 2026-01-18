import os
import tempfile
import warnings
from abc import abstractmethod
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_file, find_jar, java
from nltk.tag.api import TaggerI
@property
def _cmd(self):
    return ['edu.stanford.nlp.ie.crf.CRFClassifier', '-loadClassifier', self._stanford_model, '-textFile', self._input_file_path, '-outputFormat', self._FORMAT, '-tokenizerFactory', 'edu.stanford.nlp.process.WhitespaceTokenizer', '-tokenizerOptions', '"tokenizeNLs=false"']
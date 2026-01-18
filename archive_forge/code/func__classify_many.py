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
def _classify_many(self, featuresets, options):
    config_weka()
    temp_dir = tempfile.mkdtemp()
    try:
        test_filename = os.path.join(temp_dir, 'test.arff')
        self._formatter.write(test_filename, featuresets)
        cmd = ['weka.classifiers.bayes.NaiveBayes', '-l', self._model, '-T', test_filename] + options
        stdout, stderr = java(cmd, classpath=_weka_classpath, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if stderr and (not stdout):
            if 'Illegal options: -distribution' in stderr:
                raise ValueError('The installed version of weka does not support probability distribution output.')
            else:
                raise ValueError('Weka failed to generate output:\n%s' % stderr)
        return self.parse_weka_output(stdout.decode(stdin.encoding).split('\n'))
    finally:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
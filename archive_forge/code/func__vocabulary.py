import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def _vocabulary(self):
    with closing(data.find('stemmers/porter_test/porter_vocabulary.txt').open(encoding='utf-8')) as fp:
        return fp.read().splitlines()
import codecs
import os.path
import nltk
from nltk.chunk import tagstr2tree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import *
from nltk.tree import Tree
def _untag(self, tree):
    for i, child in enumerate(tree):
        if isinstance(child, Tree):
            self._untag(child)
        elif isinstance(child, tuple):
            tree[i] = child[0]
        else:
            raise ValueError('expected child to be Tree or tuple')
    return tree
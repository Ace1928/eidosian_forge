import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
def _classifier_builder(self, train):
    return MaxentClassifier.train(train, algorithm='megam', gaussian_prior_sigma=1, trace=2)
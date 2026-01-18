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
def _feature_detector(self, tokens, index, history):
    word = tokens[index][0]
    pos = simplify_pos(tokens[index][1])
    if index == 0:
        prevword = prevprevword = None
        prevpos = prevprevpos = None
        prevshape = prevtag = prevprevtag = None
    elif index == 1:
        prevword = tokens[index - 1][0].lower()
        prevprevword = None
        prevpos = simplify_pos(tokens[index - 1][1])
        prevprevpos = None
        prevtag = history[index - 1][0]
        prevshape = prevprevtag = None
    else:
        prevword = tokens[index - 1][0].lower()
        prevprevword = tokens[index - 2][0].lower()
        prevpos = simplify_pos(tokens[index - 1][1])
        prevprevpos = simplify_pos(tokens[index - 2][1])
        prevtag = history[index - 1]
        prevprevtag = history[index - 2]
        prevshape = shape(prevword)
    if index == len(tokens) - 1:
        nextword = nextnextword = None
        nextpos = nextnextpos = None
    elif index == len(tokens) - 2:
        nextword = tokens[index + 1][0].lower()
        nextpos = tokens[index + 1][1].lower()
        nextnextword = None
        nextnextpos = None
    else:
        nextword = tokens[index + 1][0].lower()
        nextpos = tokens[index + 1][1].lower()
        nextnextword = tokens[index + 2][0].lower()
        nextnextpos = tokens[index + 2][1].lower()
    features = {'bias': True, 'shape': shape(word), 'wordlen': len(word), 'prefix3': word[:3].lower(), 'suffix3': word[-3:].lower(), 'pos': pos, 'word': word, 'en-wordlist': word in self._english_wordlist(), 'prevtag': prevtag, 'prevpos': prevpos, 'nextpos': nextpos, 'prevword': prevword, 'nextword': nextword, 'word+nextpos': f'{word.lower()}+{nextpos}', 'pos+prevtag': f'{pos}+{prevtag}', 'shape+prevtag': f'{prevshape}+{prevtag}'}
    return features
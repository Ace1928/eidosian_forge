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
@staticmethod
def _parse_to_tagged(sent):
    """
        Convert a chunk-parse tree to a list of tagged tokens.
        """
    toks = []
    for child in sent:
        if isinstance(child, Tree):
            if len(child) == 0:
                print('Warning -- empty chunk in sentence')
                continue
            toks.append((child[0], f'B-{child.label()}'))
            for tok in child[1:]:
                toks.append((tok, f'I-{child.label()}'))
        else:
            toks.append((child, 'O'))
    return toks
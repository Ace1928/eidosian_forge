import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
from urllib.parse import unquote_plus
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
def _pos_match(pos_tuple):
    """
    This function returns the complete pos tuple for the partial pos
    tuple given to it.  It attempts to match it against the first
    non-null component of the given pos tuple.
    """
    if pos_tuple[0] == 's':
        pos_tuple = ('a', pos_tuple[1], pos_tuple[2])
    for n, x in enumerate(pos_tuple):
        if x is not None:
            break
    for pt in _pos_tuples():
        if pt[n] == pos_tuple[n]:
            return pt
    return None
import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def _utterance_fileids(self, utterances, extension):
    if utterances is None:
        utterances = self._utterances
    if isinstance(utterances, str):
        utterances = [utterances]
    return [f'{u}{extension}' for u in utterances]
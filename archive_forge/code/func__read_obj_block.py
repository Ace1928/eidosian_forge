from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
def _read_obj_block(self, stream):
    line = stream.readline()
    if line:
        return [PPAttachment(*line.split())]
    else:
        return []
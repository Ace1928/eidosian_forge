import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def _read_features(self, stream):
    features = []
    for i in range(20):
        line = stream.readline()
        if not line:
            return features
        features.extend(re.findall(FEATURES, line))
    return features
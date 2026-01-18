import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def _read_sent_block(self, stream):
    while True:
        line = stream.readline()
        if re.match(STARS, line):
            while True:
                line = stream.readline()
                if re.match(STARS, line):
                    break
            continue
        if not re.findall(COMPARISON, line) and (not ENTITIES_FEATS.findall(line)) and (not re.findall(CLOSE_COMPARISON, line)):
            if self._sent_tokenizer:
                return [self._word_tokenizer.tokenize(sent) for sent in self._sent_tokenizer.tokenize(line)]
            else:
                return [self._word_tokenizer.tokenize(line)]
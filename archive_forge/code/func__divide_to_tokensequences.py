import math
import re
from nltk.tokenize.api import TokenizerI
def _divide_to_tokensequences(self, text):
    """Divides the text into pseudosentences of fixed size"""
    w = self.w
    wrdindex_list = []
    matches = re.finditer('\\w+', text)
    for match in matches:
        wrdindex_list.append((match.group(), match.start()))
    return [TokenSequence(i / w, wrdindex_list[i:i + w]) for i in range(0, len(wrdindex_list), w)]
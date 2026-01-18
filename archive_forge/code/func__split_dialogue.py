from parlai.core.teachers import Teacher
from .build import build
import os
import random
def _split_dialogue(self, words, separator=EOS_TOKEN):
    sentences = []
    start = 0
    for stop in range(len(words)):
        if words[stop] == separator:
            sentences.append(words[start:stop])
            start = stop + 1
    if stop >= start:
        sentences.append(words[start:])
    return sentences
import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def _chunksets(t, count, chunk_label):
    pos = 0
    chunks = []
    for child in t:
        if isinstance(child, Tree):
            if re.match(chunk_label, child.label()):
                chunks.append(((count, pos), child.freeze()))
            pos += len(child.leaves())
        else:
            pos += 1
    return set(chunks)
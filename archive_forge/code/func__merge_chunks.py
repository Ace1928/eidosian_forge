import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _merge_chunks(chunks):
    if len(chunks) < 2:
        return chunks
    chunks.sort()
    start, end = chunks[0]
    new_chunks = []
    for s, e in chunks[1:]:
        if s <= end:
            end = max(end, e)
        else:
            new_chunks.append((start, end))
            start, end = (s, e)
    new_chunks.append((start, end))
    return new_chunks
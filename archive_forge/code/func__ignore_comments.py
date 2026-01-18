from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
def _ignore_comments(self, block):
    if self._comment_spans:
        while block:
            comment_end = self._find_comment_end(block)
            if comment_end != -1:
                block = block[comment_end + 1:]
                break
            else:
                block = self._read_block()
    self._comment_spans = False
    while True:
        comment_start = block.find(b'#')
        if comment_start == -1:
            break
        comment_end = self._find_comment_end(block, comment_start)
        if comment_end != -1:
            block = block[:comment_start] + block[comment_end + 1:]
        else:
            block = block[:comment_start]
            self._comment_spans = True
            break
    return block
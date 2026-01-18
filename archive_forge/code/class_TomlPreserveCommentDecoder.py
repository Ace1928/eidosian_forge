import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
class TomlPreserveCommentDecoder(TomlDecoder):

    def __init__(self, _dict=dict):
        self.saved_comments = {}
        super(TomlPreserveCommentDecoder, self).__init__(_dict)

    def preserve_comment(self, line_no, key, comment, beginline):
        self.saved_comments[line_no] = (key, comment, beginline)

    def embed_comments(self, idx, currentlevel):
        if idx not in self.saved_comments:
            return
        key, comment, beginline = self.saved_comments[idx]
        currentlevel[key] = CommentValue(currentlevel[key], comment, beginline, self._dict)
from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.compat import utf8, unichr, PY3, check_anchorname_char, nprint  # NOQA
def _gather_comments(self):
    """combine multiple comment lines"""
    comments = []
    if not self.tokens:
        return comments
    if isinstance(self.tokens[0], CommentToken):
        comment = self.tokens.pop(0)
        self.tokens_taken += 1
        comments.append(comment)
    while self.need_more_tokens():
        self.fetch_more_tokens()
        if not self.tokens:
            return comments
        if isinstance(self.tokens[0], CommentToken):
            self.tokens_taken += 1
            comment = self.tokens.pop(0)
            comments.append(comment)
    if len(comments) >= 1:
        self.tokens[0].add_pre_comments(comments)
    if not self.done and len(self.tokens) < 2:
        self.fetch_more_tokens()
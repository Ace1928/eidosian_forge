from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.compat import utf8, unichr, PY3, check_anchorname_char, nprint  # NOQA
class RoundTripScanner(Scanner):

    def check_token(self, *choices):
        while self.need_more_tokens():
            self.fetch_more_tokens()
        self._gather_comments()
        if bool(self.tokens):
            if not choices:
                return True
            for choice in choices:
                if isinstance(self.tokens[0], choice):
                    return True
        return False

    def peek_token(self):
        while self.need_more_tokens():
            self.fetch_more_tokens()
        self._gather_comments()
        if bool(self.tokens):
            return self.tokens[0]
        return None

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

    def get_token(self):
        while self.need_more_tokens():
            self.fetch_more_tokens()
        self._gather_comments()
        if bool(self.tokens):
            if len(self.tokens) > 1 and isinstance(self.tokens[0], (ScalarToken, ValueToken, FlowSequenceEndToken, FlowMappingEndToken)) and isinstance(self.tokens[1], CommentToken) and (self.tokens[0].end_mark.line == self.tokens[1].start_mark.line):
                self.tokens_taken += 1
                self.tokens[0].add_post_comment(self.tokens.pop(1))
            self.tokens_taken += 1
            return self.tokens.pop(0)
        return None

    def fetch_comment(self, comment):
        value, start_mark, end_mark = comment
        while value and value[-1] == ' ':
            value = value[:-1]
        self.tokens.append(CommentToken(value, start_mark, end_mark))

    def scan_to_next_token(self):
        srp = self.reader.peek
        srf = self.reader.forward
        if self.reader.index == 0 and srp() == '\ufeff':
            srf()
        found = False
        while not found:
            while srp() == ' ':
                srf()
            ch = srp()
            if ch == '#':
                start_mark = self.reader.get_mark()
                comment = ch
                srf()
                while ch not in _THE_END:
                    ch = srp()
                    if ch == '\x00':
                        break
                    comment += ch
                    srf()
                ch = self.scan_line_break()
                while len(ch) > 0:
                    comment += ch
                    ch = self.scan_line_break()
                end_mark = self.reader.get_mark()
                if not self.flow_level:
                    self.allow_simple_key = True
                return (comment, start_mark, end_mark)
            if bool(self.scan_line_break()):
                start_mark = self.reader.get_mark()
                if not self.flow_level:
                    self.allow_simple_key = True
                ch = srp()
                if ch == '\n':
                    start_mark = self.reader.get_mark()
                    comment = ''
                    while ch:
                        ch = self.scan_line_break(empty_line=True)
                        comment += ch
                    if srp() == '#':
                        comment = comment.rsplit('\n', 1)[0] + '\n'
                    end_mark = self.reader.get_mark()
                    return (comment, start_mark, end_mark)
            else:
                found = True
        return None

    def scan_line_break(self, empty_line=False):
        ch = self.reader.peek()
        if ch in '\r\n\x85':
            if self.reader.prefix(2) == '\r\n':
                self.reader.forward(2)
            else:
                self.reader.forward()
            return '\n'
        elif ch in '\u2028\u2029':
            self.reader.forward()
            return ch
        elif empty_line and ch in '\t ':
            self.reader.forward()
            return ch
        return ''

    def scan_block_scalar(self, style, rt=True):
        return Scanner.scan_block_scalar(self, style, rt=rt)
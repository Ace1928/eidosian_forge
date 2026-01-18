import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _code_block_with_lexer_sub(self, codeblock, leading_indent, lexer, is_fenced_code_block):
    if is_fenced_code_block:
        formatter_opts = self.extras['fenced-code-blocks'] or {}
    else:
        formatter_opts = {}

    def unhash_code(codeblock):
        for key, sanitized in list(self.html_spans.items()):
            codeblock = codeblock.replace(key, sanitized)
        replacements = [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>')]
        for old, new in replacements:
            codeblock = codeblock.replace(old, new)
        return codeblock
    _, codeblock = self._uniform_outdent(codeblock, max_outdent=leading_indent)
    codeblock = unhash_code(codeblock)
    colored = self._color_with_pygments(codeblock, lexer, **formatter_opts)
    return '\n%s\n' % self._uniform_indent(colored, leading_indent, True)
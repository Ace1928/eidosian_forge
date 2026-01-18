import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _code_block_sub(self, match, is_fenced_code_block=False):
    lexer_name = None
    if is_fenced_code_block:
        lexer_name = match.group(2)
        codeblock = match.group(3)
        codeblock = codeblock[:-1]
    else:
        codeblock = match.group(1)
        codeblock = self._outdent(codeblock)
        codeblock = self._detab(codeblock)
        codeblock = codeblock.lstrip('\n')
        codeblock = codeblock.rstrip()
    if lexer_name and 'highlightjs-lang' not in self.extras:
        lexer = self._get_pygments_lexer(lexer_name)
        if lexer:
            leading_indent = ' ' * (len(match.group(1)) - len(match.group(1).lstrip()))
            return self._code_block_with_lexer_sub(codeblock, leading_indent, lexer, is_fenced_code_block)
    pre_class_str = self._html_class_str_from_tag('pre')
    if 'highlightjs-lang' in self.extras and lexer_name:
        code_class_str = ' class="%s language-%s"' % (lexer_name, lexer_name)
    else:
        code_class_str = self._html_class_str_from_tag('code')
    if is_fenced_code_block:
        leading_indent = ' ' * (len(match.group(1)) - len(match.group(1).lstrip()))
        if codeblock:
            leading_indent, codeblock = self._uniform_outdent(codeblock, max_outdent=leading_indent)
        codeblock = self._encode_code(codeblock)
        if lexer_name == 'mermaid' and 'mermaid' in self.extras:
            return '\n%s<pre class="mermaid-pre"><div class="mermaid">%s\n</div></pre>\n' % (leading_indent, codeblock)
        return '\n%s<pre%s><code%s>%s\n</code></pre>\n' % (leading_indent, pre_class_str, code_class_str, codeblock)
    else:
        codeblock = self._encode_code(codeblock)
        return '\n<pre%s><code%s>%s\n</code></pre>\n' % (pre_class_str, code_class_str, codeblock)
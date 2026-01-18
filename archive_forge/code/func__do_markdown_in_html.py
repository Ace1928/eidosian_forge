import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_markdown_in_html(self, text):

    def callback(block):
        indent, block = self._uniform_outdent(block)
        block = self._hash_html_block_sub(block)
        block = self._uniform_indent(block, indent, include_empty_lines=True, indent_empty_lines=False)
        return block
    return self._strict_tag_block_sub(text, self._block_tags_a, callback, True)
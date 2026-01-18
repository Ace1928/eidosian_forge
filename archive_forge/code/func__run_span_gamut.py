import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _run_span_gamut(self, text):
    text = self._do_code_spans(text)
    text = self._escape_special_chars(text)
    if 'link-patterns' in self.extras:
        text = self._do_link_patterns(text)
    text = self._do_links(text)
    text = self._do_auto_links(text)
    text = self._encode_amps_and_angles(text)
    if 'strike' in self.extras:
        text = self._do_strike(text)
    if 'underline' in self.extras:
        text = self._do_underline(text)
    text = self._do_italics_and_bold(text)
    if 'tg-spoiler' in self.extras:
        text = self._do_tg_spoiler(text)
    if 'smarty-pants' in self.extras:
        text = self._do_smart_punctuation(text)
    on_backslash = self.extras.get('breaks', {}).get('on_backslash', False)
    on_newline = self.extras.get('breaks', {}).get('on_newline', False)
    if on_backslash and on_newline:
        pattern = ' *\\\\?'
    elif on_backslash:
        pattern = '(?: *\\\\| {2,})'
    elif on_newline:
        pattern = ' *'
    else:
        pattern = ' {2,}'
    break_tag = '<br%s\n' % self.empty_element_suffix
    text = re.sub(pattern + '\\n(?!\\<(?:\\/?(ul|ol|li))\\>)', break_tag, text)
    return text
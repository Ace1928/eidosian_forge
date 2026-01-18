import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _wavedrom_block_sub(self, match):
    if match.group(2) != 'wavedrom':
        return match.string[match.start():match.end()]
    lead_indent, waves = self._uniform_outdent(match.group(3))
    open_tag, close_tag = ('<script type="WaveDrom">\n', '</script>')
    if not isinstance(self.extras['wavedrom'], dict):
        embed_svg = True
    else:
        embed_svg = self.extras['wavedrom'].get('prefer_embed_svg', True)
    if embed_svg:
        try:
            import wavedrom
            waves = wavedrom.render(waves).tostring()
            open_tag, close_tag = ('<div>', '\n</div>')
        except ImportError:
            pass
    self._escape_table[waves] = _hash_text(waves)
    return self._uniform_indent('\n%s%s%s\n' % (open_tag, self._escape_table[waves], close_tag), lead_indent, indent_empty_lines=True)
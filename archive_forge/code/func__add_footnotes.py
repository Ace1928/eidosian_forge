import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _add_footnotes(self, text):
    if self.footnotes:
        footer = ['<div class="footnotes">', '<hr' + self.empty_element_suffix, '<ol>']
        if not self.footnote_title:
            self.footnote_title = 'Jump back to footnote %d in the text.'
        if not self.footnote_return_symbol:
            self.footnote_return_symbol = '&#8617;'
        self.footnote_ids.sort(key=lambda a: list(self.footnotes.keys()).index(a))
        for i, id in enumerate(self.footnote_ids):
            if i != 0:
                footer.append('')
            footer.append('<li id="fn-%s">' % id)
            footer.append(self._run_block_gamut(self.footnotes[id]))
            try:
                backlink = ('<a href="#fnref-%s" ' + 'class="footnoteBackLink" ' + 'title="' + self.footnote_title + '">' + self.footnote_return_symbol + '</a>') % (id, i + 1)
            except TypeError:
                log.debug('Footnote error. `footnote_title` must include parameter. Using defaults.')
                backlink = '<a href="#fnref-%s" class="footnoteBackLink" title="Jump back to footnote %d in the text.">&#8617;</a>' % (id, i + 1)
            if footer[-1].endswith('</p>'):
                footer[-1] = footer[-1][:-len('</p>')] + '&#160;' + backlink + '</p>'
            else:
                footer.append('\n<p>%s</p>' % backlink)
            footer.append('</li>')
        footer.append('</ol>')
        footer.append('</div>')
        return text + '\n\n' + '\n'.join(footer)
    else:
        return text
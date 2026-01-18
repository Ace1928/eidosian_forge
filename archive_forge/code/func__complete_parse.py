import re
import sys
import copy
import unicodedata
import reportlab.lib.sequencer
from reportlab.lib.abag import ABag
from reportlab.lib.utils import ImageReader, annotateException, encode_label, asUnicode
from reportlab.lib.colors import toColor, black
from reportlab.lib.fonts import tt2ps, ps2tt
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch,mm,cm,pica
from reportlab.rl_config import platypus_link_underline
from html.parser import HTMLParser
from html.entities import name2codepoint
def _complete_parse(self):
    """Reset after parsing, to be ready for next paragraph"""
    if self._stack:
        self._syntax_error('parse ended with %d unclosed tags\n %s' % (len(self._stack), '\n '.join((x.__tag__ for x in reversed(self._stack)))))
    del self._seq
    style = self._style
    del self._style
    if len(self.errors) == 0:
        fragList = self.fragList
        bFragList = hasattr(self, 'bFragList') and self.bFragList or None
        self._iReset()
    else:
        fragList = bFragList = None
    return (style, fragList, bFragList)
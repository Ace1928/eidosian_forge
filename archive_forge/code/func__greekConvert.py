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
def _greekConvert(data):
    global _greek2Utf8
    if not _greek2Utf8:
        from reportlab.pdfbase.rl_codecs import RL_Codecs
        import codecs
        dm = codecs.make_identity_dict(range(32, 256))
        for k in range(0, 32):
            dm[k] = None
        dm.update(RL_Codecs._RL_Codecs__rl_codecs_data['symbol'][0])
        _greek2Utf8 = {}
        for k, v in dm.items():
            if not v:
                u = '\x00'
            else:
                u = chr(v)
            _greek2Utf8[chr(k)] = u
    return ''.join(map(_greek2Utf8.__getitem__, data))
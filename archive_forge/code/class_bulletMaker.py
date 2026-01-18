from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class bulletMaker:

    def __init__(self, tagname, atts, context):
        self.tagname = tagname
        style = 'li.defaultStyle'
        self.style = style = atts.get('style', style)
        typ = {'ul': 'disc', 'ol': '1', 'dl': None}[tagname]
        self.typ = typ = atts.get('type', typ)
        if 'leftIndent' not in atts:
            thestyle = context[style]
            from reportlab.pdfbase.pdfmetrics import stringWidth
            size = thestyle.fontSize
            indent = stringWidth('XXX', 'Courier', size)
            atts['leftIndent'] = str(indent)
        self.count = 0
        self._first = 1

    def makeBullet(self, atts, bl=None):
        if not self._first:
            atts['spaceBefore'] = '0'
        else:
            self._first = 0
        typ = self.typ
        tagname = self.tagname
        if bl is None:
            if tagname == 'ul':
                if typ == 'disc':
                    bl = chr(109)
                elif typ == 'circle':
                    bl = chr(108)
                elif typ == 'square':
                    bl = chr(110)
                else:
                    raise ValueError('unordered list type %s not implemented' % repr(typ))
                if 'bulletFontName' not in atts:
                    atts['bulletFontName'] = 'ZapfDingbats'
            elif tagname == 'ol':
                if 'value' in atts:
                    self.count = int(atts['value'])
                else:
                    self.count += 1
                if typ == '1':
                    bl = str(self.count)
                elif typ == 'a':
                    theord = ord('a') + self.count - 1
                    bl = chr(theord)
                elif typ == 'A':
                    theord = ord('A') + self.count - 1
                    bl = chr(theord)
                else:
                    raise ValueError('ordered bullet type %s not implemented' % repr(typ))
            else:
                raise ValueError('bad tagname ' + repr(tagname))
        if 'bulletText' not in atts:
            atts['bulletText'] = bl
        if 'style' not in atts:
            atts['style'] = self.style
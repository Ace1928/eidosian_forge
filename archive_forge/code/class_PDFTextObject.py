from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
class PDFTextObject(_PDFColorSetter):
    """PDF logically separates text and graphics drawing; text
    operations need to be bracketed between BT (Begin text) and
    ET operators. This class ensures text operations are
    properly encapusalted. Ask the canvas for a text object
    with beginText(x, y).  Do not construct one directly.
    Do not use multiple text objects in parallel; PDF is
    not multi-threaded!

    It keeps track of x and y coordinates relative to its origin."""

    def __init__(self, canvas, x=0, y=0, direction=None):
        self._code = ['BT']
        self._canvas = canvas
        self._fontname = self._canvas._fontname
        self._fontsize = self._canvas._fontsize
        self._leading = self._canvas._leading
        self._doc = self._canvas._doc
        self._colorsUsed = self._canvas._colorsUsed
        self._enforceColorSpace = getattr(canvas, '_enforceColorSpace', None)
        font = pdfmetrics.getFont(self._fontname)
        self._curSubset = -1
        self.direction = direction
        self.setTextOrigin(x, y)
        self._textRenderMode = 0
        self._clipping = 0

    def getCode(self):
        """pack onto one line; used internally"""
        self._code.append('ET')
        if self._clipping:
            self._code.append('%d Tr' % (self._textRenderMode ^ 4))
        return ' '.join(self._code)

    def setTextOrigin(self, x, y):
        if self._canvas.bottomup:
            self._code.append('1 0 0 1 %s Tm' % fp_str(x, y))
        else:
            self._code.append('1 0 0 -1 %s Tm' % fp_str(x, y))
        self._x0 = self._x = x
        self._y0 = self._y = y

    def setTextTransform(self, a, b, c, d, e, f):
        """Like setTextOrigin, but does rotation, scaling etc."""
        if not self._canvas.bottomup:
            c = -c
            d = -d
        self._code.append('%s Tm' % fp_str(a, b, c, d, e, f))
        self._x0 = self._x = e
        self._y0 = self._y = f

    def moveCursor(self, dx, dy):
        """Starts a new line at an offset dx,dy from the start of the
        current line. This does not move the cursor relative to the
        current position, and it changes the current offset of every
        future line drawn (i.e. if you next do a textLine() call, it
        will move the cursor to a position one line lower than the
        position specificied in this call.  """
        if self._code and self._code[-1][-3:] == ' Td':
            L = self._code[-1].split()
            if len(L) == 3:
                del self._code[-1]
            else:
                self._code[-1] = ''.join(L[:-4])
            lastDx = float(L[-3])
            lastDy = float(L[-2])
            dx += lastDx
            dy -= lastDy
            self._x0 -= lastDx
            self._y0 -= lastDy
        self._code.append('%s Td' % fp_str(dx, -dy))
        self._x0 += dx
        self._y0 += dy
        self._x = self._x0
        self._y = self._y0

    def setXPos(self, dx):
        """Starts a new line dx away from the start of the
        current line - NOT from the current point! So if
        you call it in mid-sentence, watch out."""
        self.moveCursor(dx, 0)

    def getCursor(self):
        """Returns current text position relative to the last origin."""
        return (self._x, self._y)

    def getStartOfLine(self):
        """Returns a tuple giving the text position of the start of the
        current line."""
        return (self._x0, self._y0)

    def getX(self):
        """Returns current x position relative to the last origin."""
        return self._x

    def getY(self):
        """Returns current y position relative to the last origin."""
        return self._y

    def _setFont(self, psfontname, size):
        """Sets the font and fontSize
        Raises a readable exception if an illegal font
        is supplied.  Font names are case-sensitive! Keeps track
        of font anme and size for metrics."""
        self._fontname = psfontname
        self._fontsize = size
        font = pdfmetrics.getFont(self._fontname)
        if font._dynamicFont:
            self._curSubset = -1
        else:
            pdffontname = self._canvas._doc.getInternalFontName(psfontname)
            self._code.append('%s %s Tf' % (pdffontname, fp_str(size)))

    def setFont(self, psfontname, size, leading=None):
        """Sets the font.  If leading not specified, defaults to 1.2 x
        font size. Raises a readable exception if an illegal font
        is supplied.  Font names are case-sensitive! Keeps track
        of font anme and size for metrics."""
        self._fontname = psfontname
        self._fontsize = size
        if leading is None:
            leading = size * 1.2
        self._leading = leading
        font = pdfmetrics.getFont(self._fontname)
        if font._dynamicFont:
            self._curSubset = -1
        else:
            pdffontname = self._canvas._doc.getInternalFontName(psfontname)
            self._code.append('%s %s Tf %s TL' % (pdffontname, fp_str(size), fp_str(leading)))

    def setCharSpace(self, charSpace):
        """Adjusts inter-character spacing"""
        self._charSpace = charSpace
        self._code.append('%s Tc' % fp_str(charSpace))

    def setWordSpace(self, wordSpace):
        """Adjust inter-word spacing.  This can be used
        to flush-justify text - you get the width of the
        words, and add some space between them."""
        self._wordSpace = wordSpace
        self._code.append('%s Tw' % fp_str(wordSpace))

    def setHorizScale(self, horizScale):
        """Stretches text out horizontally"""
        self._horizScale = 100 + horizScale
        self._code.append('%s Tz' % fp_str(horizScale))

    def setLeading(self, leading):
        """How far to move down at the end of a line."""
        self._leading = leading
        self._code.append('%s TL' % fp_str(leading))

    def setTextRenderMode(self, mode):
        """Set the text rendering mode.

        0 = Fill text
        1 = Stroke text
        2 = Fill then stroke
        3 = Invisible
        4 = Fill text and add to clipping path
        5 = Stroke text and add to clipping path
        6 = Fill then stroke and add to clipping path
        7 = Add to clipping path

        after we start clipping we mustn't change the mode back until after the ET
        """
        assert mode in (0, 1, 2, 3, 4, 5, 6, 7), 'mode must be in (0,1,2,3,4,5,6,7)'
        if mode & 4 != self._clipping:
            mode |= 4
            self._clipping = mode & 4
        if self._textRenderMode != mode:
            self._textRenderMode = mode
            self._code.append('%d Tr' % mode)

    def setRise(self, rise):
        """Move text baseline up or down to allow superscript/subscripts"""
        self._rise = rise
        self._y = self._y - rise
        self._code.append('%s Ts' % fp_str(rise))

    def _formatText(self, text):
        """Generates PDF text output operator(s)"""
        if log2vis and self.direction in ('LTR', 'RTL'):
            text = log2vis(text, directionsMap.get(self.direction, DIR_ON), clean=True)
        canv = self._canvas
        font = pdfmetrics.getFont(self._fontname)
        R = []
        if font._dynamicFont:
            for subset, t in font.splitString(text, canv._doc):
                if subset != self._curSubset:
                    pdffontname = font.getSubsetInternalName(subset, canv._doc)
                    R.append('%s %s Tf %s TL' % (pdffontname, fp_str(self._fontsize), fp_str(self._leading)))
                    self._curSubset = subset
                R.append('(%s) Tj' % canv._escape(t))
        elif font._multiByte:
            R.append('%s %s Tf %s TL' % (canv._doc.getInternalFontName(font.fontName), fp_str(self._fontsize), fp_str(self._leading)))
            R.append('(%s) Tj' % font.formatForPdf(text))
        else:
            fc = font
            if isBytes(text):
                try:
                    text = text.decode('utf8')
                except UnicodeDecodeError as e:
                    i, j = e.args[2:4]
                    raise UnicodeDecodeError(*e.args[:4] + ('%s\n%s-->%s<--%s' % (e.args[4], text[max(i - 10, 0):i], text[i:j], text[j:j + 10]),))
            for f, t in pdfmetrics.unicode2T1(text, [font] + font.substitutionFonts):
                if f != fc:
                    R.append('%s %s Tf %s TL' % (canv._doc.getInternalFontName(f.fontName), fp_str(self._fontsize), fp_str(self._leading)))
                    fc = f
                R.append('(%s) Tj' % canv._escape(t))
            if font != fc:
                R.append('%s %s Tf %s TL' % (canv._doc.getInternalFontName(self._fontname), fp_str(self._fontsize), fp_str(self._leading)))
        return ' '.join(R)

    def _textOut(self, text, TStar=0):
        """prints string at current point, ignores text cursor"""
        self._code.append('%s%s' % (self._formatText(text), TStar and ' T*' or ''))

    def textOut(self, text):
        """prints string at current point, text cursor moves across."""
        self._x = self._x + self._canvas.stringWidth(text, self._fontname, self._fontsize)
        self._code.append(self._formatText(text))

    def textLine(self, text=''):
        """prints string at current point, text cursor moves down.
        Can work with no argument to simply move the cursor down."""
        self._x = self._x0
        if self._canvas.bottomup:
            self._y = self._y - self._leading
        else:
            self._y = self._y + self._leading
        self._y0 = self._y
        self._code.append('%s T*' % self._formatText(text))

    def textLines(self, stuff, trim=1):
        """prints multi-line or newlined strings, moving down.  One
        comon use is to quote a multi-line block in your Python code;
        since this may be indented, by default it trims whitespace
        off each line and from the beginning; set trim=0 to preserve
        whitespace."""
        if isStr(stuff):
            lines = asUnicode(stuff).strip().split(u'\n')
            if trim == 1:
                lines = [s.strip() for s in lines]
        elif isinstance(stuff, (tuple, list)):
            lines = stuff
        else:
            assert 1 == 0, 'argument to textlines must be string,, list or tuple'
        for line in lines:
            self.textLine(line)

    def __nonzero__(self):
        """PDFTextObject is true if it has something done after the init"""
        return self._code != ['BT']

    def _setFillAlpha(self, v):
        self._canvas._doc.ensureMinPdfVersion('transparency')
        self._canvas._extgstate.set(self, 'ca', v)

    def _setStrokeOverprint(self, v):
        self._canvas._extgstate.set(self, 'OP', v)

    def _setFillOverprint(self, v):
        self._canvas._extgstate.set(self, 'op', v)

    def _setOverprintMask(self, v):
        self._canvas._extgstate.set(self, 'OPM', v and 1 or 0)
import formatter
import string
from types import *
import htmllib
import piddle
class _HtmlPiddleWriter:
    FontSizeDict = {'h1': 36, 'h2': 24, 'h3': 18, 'h4': 12, 'h5': 10, 'h6': 8}
    DefaultFontSize = 12

    def __init__(self, aHTMLPiddler, aPiddleCanvas):
        self.piddler = aHTMLPiddler
        self.pc = aPiddleCanvas
        self.anchor = None
        self.lineHeight = 0
        self.atbreak = 0
        self.color = self.piddler.color
        self.defaultFont = self.font = self.piddler.font
        s = 'W' * 20
        x = self.pc.stringWidth(s, self.font)
        y = self.pc.fontHeight(self.font)
        x = (x + 19) / 20
        self.fsizex = x
        self.fsizey = self.oldLineHeight = y
        self.indentSize = x * 3
        self.lmargin, self.rmargin = self.piddler.xLimits
        self.x, self.y = self.piddler.start
        self.indent = self.lmargin + x / 3

    def anchor_bgn(self, href, name, type):
        if href:
            self.oldcolor = self.color
            self.color = piddle.Color(0.0, 0.0, 200 / 255.0)
            self.anchor = (href, name, type)

    def anchor_end(self):
        if self.anchor:
            self.color = self.oldcolor
            self.anchor = None

    def new_font(self, fontParams):
        if TRACE:
            print('nf', fontParams)
        if not fontParams:
            fontParams = (None, None, None, None)
        size = fontParams[0]
        try:
            points = self.FontSizeDict[size]
        except KeyError:
            points = self.DefaultFontSize
        if fontParams[3]:
            face = 'courier'
        elif isinstance(size, str) and size[0] == 'h':
            face = 'helvetica'
        else:
            face = 'times'
        italic = fontParams[1]
        if italic is None:
            italic = 0
        bold = fontParams[2]
        if bold is None:
            bold = 0
        self.font = piddle.Font(points, bold, italic, face=face)
        x = self.pc.stringWidth('W' * 20, self.font)
        self.fsizex = (x + 19) / 20
        self.fsizey = self.pc.fontHeight(self.font)

    def new_margin(self, margin, level):
        self.send_line_break()
        self.indent = self.x = self.lmargin + self.indentSize * level

    def new_spacing(self, spacing):
        self.send_line_break()
        t = 'new_spacing(%s)' % repr(spacing)
        self.OutputLine(t, 1)

    def new_styles(self, styles):
        self.send_line_break()
        t = 'new_styles(%s)' % repr(styles)
        self.OutputLine(t, 1)

    def send_label_data(self, data):
        if data == '*':
            w = self.pc.stringWidth(data, self.font) / 3
            h = self.pc.fontHeight(self.font) / 3
            x = self.indent - w
            y = self.y - w
            self.pc.drawRect(x, y, x - w, y - w)
        else:
            w = self.pc.stringWidth(data, self.font)
            h = self.pc.fontHeight(self.font)
            x = self.indent - w - self.fsizex / 3
            if x < 0:
                x = 0
            self.pc.drawString(data, x, self.y, self.font, self.color)

    def send_paragraph(self, blankline):
        self.send_line_break()
        self.y = self.y + self.oldLineHeight * blankline

    def send_line_break(self):
        if self.lineHeight:
            self.y = self.y + self.lineHeight
            self.oldLineHeight = self.lineHeight
            self.lineHeight = 0
        self.x = self.indent
        self.atbreak = 0
        if TRACE:
            input('lb')

    def send_hor_rule(self):
        self.send_line_break()
        self.y = self.y + self.oldLineHeight
        border = self.fsizex
        self.pc.drawLine(border, self.y, self.rmargin - border, self.y, piddle.Color(0.0, 0.0, 200 / 255.0))
        self.y = self.y + self.oldLineHeight

    def send_literal_data(self, data):
        if not data:
            return
        lines = data.split(data, '\n')
        text = lines[0].replace('\t', ' ' * 8)
        for l in lines[1:]:
            self.OutputLine(text, 1)
            text = l.replace('\t', ' ' * 8)
        self.OutputLine(text, 0)
        self.atbreak = 0

    def send_flowing_data(self, data):
        if not data:
            return
        atbreak = self.atbreak or data[0] in string.whitespace
        text = ''
        pixels = chars = 0
        for word in data.split():
            bword = ' ' + word
            length = len(bword)
            if not atbreak:
                text = word
                chars = chars + length - 1
            elif self.x + pixels + (chars + length) * self.fsizex < self.rmargin:
                text = text + bword
                chars = chars + length
            else:
                w = self.pc.stringWidth(text + bword, self.font)
                h = self.pc.fontHeight(self.font)
                if TRACE:
                    print('sfd T:', text + bword)
                if TRACE:
                    print('sfd', self.x, w, self.x + w, self.rmargin)
                if self.x + w < self.rmargin:
                    text = text + bword
                    pixels = w
                    chars = 0
                else:
                    self.OutputLine(text, 1)
                    text = word
                    chars = length - 1
                    pixels = 0
            atbreak = 1
        self.OutputLine(text, 0)
        self.atbreak = data[-1] in string.whitespace

    def OutputLine(self, text, linebreak=0):
        if text:
            if TRACE:
                print('olt:', text)
            if TRACE:
                print('olf:', self.font.size, self.font.bold, self.font.italic, self.font.underline, self.font.face)
            self.pc.drawString(text, self.x, self.y, self.font, self.color)
            self.lineHeight = max(self.lineHeight, self.pc.fontHeight(self.font))
            self.x = self.x + self.pc.stringWidth(text, self.font)
        if linebreak:
            self.send_line_break()
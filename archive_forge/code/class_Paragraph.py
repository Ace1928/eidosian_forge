from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
class Paragraph(Flowable):
    """ Paragraph(text, style, bulletText=None, caseSensitive=1)
        text a string of stuff to go into the paragraph.
        style is a style definition as in reportlab.lib.styles.
        bulletText is an optional bullet defintion.
        caseSensitive set this to 0 if you want the markup tags and their attributes to be case-insensitive.

        This class is a flowable that can format a block of text
        into a paragraph with a given style.

        The paragraph Text can contain XML-like markup including the tags:
        <b> ... </b> - bold
        < u [color="red"] [width="pts"] [offset="pts"]> < /u > - underline
            width and offset can be empty meaning use existing canvas line width
            or with an f/F suffix regarded as a fraction of the font size
        < strike > < /strike > - strike through has the same parameters as underline
        <i> ... </i> - italics
        <u> ... </u> - underline
        <strike> ... </strike> - strike through
        <super> ... </super> - superscript
        <sub> ... </sub> - subscript
        <font name=fontfamily/fontname color=colorname size=float>
        <span name=fontfamily/fontname color=colorname backcolor=colorname size=float style=stylename>
        <onDraw name=callable label="a label"/>
        <index [name="callablecanvasattribute"] label="a label"/>
        <link>link text</link>
            attributes of links
                size/fontSize/uwidth/uoffset=num
                name/face/fontName=name
                fg/textColor/color/ucolor=color
                backcolor/backColor/bgcolor=color
                dest/destination/target/href/link=target
                underline=bool turn on underline
        <a>anchor text</a>
            attributes of anchors
                size/fontSize/uwidth/uoffset=num
                fontName=name
                fg/textColor/color/ucolor=color
                backcolor/backColor/bgcolor=color
                href=href
                underline="yes|no"
        <a name="anchorpoint"/>
        <unichar name="unicode character name"/>
        <unichar value="unicode code point"/>
        <img src="path" width="1in" height="1in" valign="bottom"/>
                width="w%" --> fontSize*w/100   idea from Roberto Alsina
                height="h%" --> linewidth*h/100 <ralsina@netmanagers.com.ar>

        The whole may be surrounded by <para> </para> tags

        The <b> and <i> tags will work for the built-in fonts (Helvetica
        /Times / Courier).  For other fonts you need to register a family
        of 4 fonts using reportlab.pdfbase.pdfmetrics.registerFont; then
        use the addMapping function to tell the library that these 4 fonts
        form a family e.g.
        from reportlab.lib.fonts import addMapping
        addMapping('Vera', 0, 0, 'Vera')    #normal
        addMapping('Vera', 0, 1, 'Vera-Italic')    #italic
        addMapping('Vera', 1, 0, 'Vera-Bold')    #bold
        addMapping('Vera', 1, 1, 'Vera-BoldItalic')    #italic and bold

        It will also be able to handle any MathML specified Greek characters.
    """

    def __init__(self, text, style=None, bulletText=None, frags=None, caseSensitive=1, encoding='utf8'):
        if style is None:
            style = ParagraphStyle(name='paragraphImplicitDefaultStyle')
        self.caseSensitive = caseSensitive
        self.encoding = encoding
        self._setup(text, style, bulletText or getattr(style, 'bulletText', None), frags, cleanBlockQuotedText)

    def __repr__(self):
        n = self.__class__.__name__
        L = [n + '(']
        keys = list(self.__dict__.keys())
        for k in keys:
            L.append('%s: %s' % (repr(k).replace('\n', ' ').replace('  ', ' '), repr(getattr(self, k)).replace('\n', ' ').replace('  ', ' ')))
        L.append(') #' + n)
        return '\n'.join(L)

    def _setup(self, text, style, bulletText, frags, cleaner):
        if frags is None:
            text = cleaner(text)
            _parser = ParaParser()
            _parser.caseSensitive = self.caseSensitive
            style, frags, bulletTextFrags = _parser.parse(text, style)
            if frags is None:
                raise ValueError("xml parser error (%s) in paragraph beginning\n'%s'" % (_parser.errors[0], text[:min(30, len(text))]))
            textTransformFrags(frags, style)
            if bulletTextFrags:
                bulletText = bulletTextFrags
        self.text = text
        self.frags = frags
        self.style = style
        self.bulletText = bulletText
        self.debug = 0

    def wrap(self, availWidth, availHeight):
        if availWidth < _FUZZ:
            return (0, 2147483647)
        self.width = availWidth
        style = self.style
        leftIndent = style.leftIndent
        first_line_width = availWidth - (leftIndent + style.firstLineIndent) - style.rightIndent
        later_widths = availWidth - leftIndent - style.rightIndent
        self._wrapWidths = [first_line_width, later_widths]
        if style.wordWrap == 'CJK':
            blPara = self.breakLinesCJK(self._wrapWidths)
        else:
            blPara = self.breakLines(self._wrapWidths)
        self.blPara = blPara
        autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
        leading = style.leading
        if blPara.kind == 1:
            if autoLeading not in ('', 'off'):
                height = 0
                if autoLeading == 'max':
                    for l in blPara.lines:
                        height += max(l.ascent - l.descent, leading)
                elif autoLeading == 'min':
                    for l in blPara.lines:
                        height += l.ascent - l.descent
                else:
                    raise ValueError('invalid autoLeading value %r' % autoLeading)
            else:
                height = len(blPara.lines) * leading
        else:
            if autoLeading == 'max':
                leading = max(leading, blPara.ascent - blPara.descent)
            elif autoLeading == 'min':
                leading = blPara.ascent - blPara.descent
            height = len(blPara.lines) * leading
        self.height = height
        return (self.width, height)

    def minWidth(self):
        """Attempt to determine a minimum sensible width"""
        frags = self.frags
        nFrags = len(frags)
        if not nFrags:
            return 0
        if nFrags == 1 and (not _processed_frags(frags)):
            f = frags[0]
            fS = f.fontSize
            fN = f.fontName
            return max((stringWidth(w, fN, fS) for w in (split(f.text, ' ') if hasattr(f, 'text') else f.words)))
        else:
            return max((w[0] for w in _getFragWords(frags)))

    def _split_blParaProcessed(self, blPara, start, stop):
        if not stop:
            return []
        lines = blPara.lines
        sFW = lines[start].sFW
        sFWN = lines[stop].sFW if stop != len(lines) else len(self.frags)
        F = self.frags[sFW:sFWN]
        while F and isinstance(F[-1], _InjectedFrag):
            del F[-1]
        if isinstance(F[-1], _SplitFragHY):
            F[-1].__class__ = _SHYWordHS if isinstance(F[-1], _SHYWord) else _SplitFragLL
        return F

    def _get_split_blParaFunc(self):
        return _split_blParaSimple if self.blPara.kind == 0 else _split_blParaHard if not _processed_frags(self.frags) else self._split_blParaProcessed

    def split(self, availWidth, availHeight):
        if len(self.frags) <= 0 or availWidth < _FUZZ or availHeight < _FUZZ:
            return []
        if not hasattr(self, 'blPara'):
            self.wrap(availWidth, availHeight)
        blPara = self.blPara
        style = self.style
        autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
        leading = style.leading
        lines = blPara.lines
        if blPara.kind == 1 and autoLeading not in ('', 'off'):
            s = height = 0
            if autoLeading == 'max':
                for i, l in enumerate(blPara.lines):
                    h = max(l.ascent - l.descent, leading)
                    n = height + h
                    if n > availHeight + 1e-08:
                        break
                    height = n
                    s = i + 1
            elif autoLeading == 'min':
                for i, l in enumerate(blPara.lines):
                    n = height + l.ascent - l.descent
                    if n > availHeight + 1e-08:
                        break
                    height = n
                    s = i + 1
            else:
                raise ValueError('invalid autoLeading value %r' % autoLeading)
        else:
            l = leading
            if autoLeading == 'max':
                l = max(leading, 1.2 * style.fontSize)
            elif autoLeading == 'min':
                l = 1.2 * style.fontSize
            s = int(availHeight / float(l))
            height = s * l
        allowOrphans = getattr(self, 'allowOrphans', getattr(style, 'allowOrphans', 0))
        if not allowOrphans and s <= 1 or s == 0:
            del self.blPara
            return []
        n = len(lines)
        allowWidows = getattr(self, 'allowWidows', getattr(style, 'allowWidows', 1))
        if n <= s:
            return [self]
        if not allowWidows:
            if n == s + 1:
                if allowOrphans and n == 3 or n > 3:
                    s -= 1
                else:
                    del self.blPara
                    return []
        func = self._get_split_blParaFunc()
        if style.endDots:
            style1 = deepcopy(style)
            style1.endDots = None
        else:
            style1 = style
        P1 = self.__class__(None, style1, bulletText=self.bulletText, frags=func(blPara, 0, s))
        P1.blPara = ParaLines(kind=1, lines=blPara.lines[0:s], aH=availHeight, aW=availWidth)
        P1._JustifyLast = not (isinstance(blPara.lines[s - 1], FragLine) and hasattr(blPara.lines[s - 1], 'lineBreak') and blPara.lines[s - 1].lineBreak)
        P1._splitpara = 1
        P1.height = height
        P1.width = availWidth
        if style.firstLineIndent != 0:
            style = deepcopy(style)
            style.firstLineIndent = 0
        P2 = self.__class__(None, style, bulletText=None, frags=func(blPara, s, n))
        for a in ('autoLeading',):
            if hasattr(self, a):
                setattr(P1, a, getattr(self, a))
                setattr(P2, a, getattr(self, a))
        return [P1, P2]

    def draw(self):
        self.drawPara(self.debug)

    def breakLines(self, width):
        """
        Returns a broken line structure. There are two cases

        A) For the simple case of a single formatting input fragment the output is
            A fragment specifier with
                - kind = 0
                - fontName, fontSize, leading, textColor
                - lines=  A list of lines

                        Each line has two items.

                        1. unused width in points
                        2. word list

        B) When there is more than one input formatting fragment the output is
            A fragment specifier with
               - kind = 1
               - lines=  A list of fragments each having fields
                            - extraspace (needed for justified)
                            - fontSize
                            - words=word list
                                each word is itself a fragment with
                                various settings
            in addition frags becomes a frag word list

        This structure can be used to easily draw paragraphs with the various alignments.
        You can supply either a single width or a list of widths; the latter will have its
        last item repeated until necessary. A 2-element list is useful when there is a
        different first line indent; a longer list could be created to facilitate custom wraps
        around irregular objects."""
        self._width_max = 0
        if not isinstance(width, (tuple, list)):
            maxWidths = [width]
        else:
            maxWidths = width
        lines = []
        self.height = lineno = 0
        maxlineno = len(maxWidths) - 1
        style = self.style
        hyphenator = getattr(style, 'hyphenationLang', '')
        if hyphenator:
            if isStr(hyphenator):
                hyphenator = hyphenator.strip()
                if hyphenator and pyphen:
                    hyphenator = pyphen.Pyphen(lang=hyphenator).iterate
                else:
                    hyphenator = None
            elif not callable(hyphenator):
                raise ValueError('hyphenator should be a language spec or a callable unicode -->  pairs not %r' % hyphenator)
        else:
            hyphenator = None
        uriWasteReduce = style.uriWasteReduce
        embeddedHyphenation = style.embeddedHyphenation
        hyphenation2 = embeddedHyphenation > 1
        spaceShrinkage = style.spaceShrinkage
        splitLongWords = style.splitLongWords
        attemptHyphenation = hyphenator or uriWasteReduce or embeddedHyphenation
        if attemptHyphenation:
            hymwl = getattr(style, 'hyphenationMinWordLength', hyphenationMinWordLength)
        self._splitLongWordCount = self._hyphenations = 0
        _handleBulletWidth(self.bulletText, style, maxWidths)
        maxWidth = maxWidths[0]
        autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
        calcBounds = autoLeading not in ('', 'off')
        frags = self.frags
        nFrags = len(frags)
        if nFrags == 1 and (not (style.endDots or hasattr(frags[0], 'cbDefn') or hasattr(frags[0], 'backColor') or _processed_frags(frags))):
            f = frags[0]
            fontSize = f.fontSize
            fontName = f.fontName
            ascent, descent = getAscentDescent(fontName, fontSize)
            if hasattr(f, 'text'):
                text = strip(f.text)
                if not text:
                    return f.clone(kind=0, lines=[], ascent=ascent, descent=descent, fontSize=fontSize)
                else:
                    words = split(text)
            else:
                words = f.words[:]
                for w in words:
                    if strip(w):
                        break
                else:
                    return f.clone(kind=0, lines=[], ascent=ascent, descent=descent, fontSize=fontSize)
            spaceWidth = stringWidth(' ', fontName, fontSize, self.encoding)
            dSpaceShrink = spaceShrinkage * spaceWidth
            cLine = []
            currentWidth = -spaceWidth
            hyw = stringWidth('-', fontName, fontSize, self.encoding)
            forcedSplit = 0
            while words:
                word = words.pop(0)
                if not word and isinstance(word, _SplitWord):
                    forcedSplit = 1
                elif _shy in word:
                    word = _SHYStr(word)
                wordWidth = stringWidth(word, fontName, fontSize, self.encoding)
                newWidth = currentWidth + spaceWidth + wordWidth
                limWidth = maxWidth + dSpaceShrink * len(cLine)
                if newWidth > limWidth and (not (isinstance(word, _SplitWordH) or forcedSplit)):
                    if isinstance(word, _SHYStr):
                        hsw = word.__shysplit__(fontName, fontSize, currentWidth + spaceWidth + hyw - 1e-08, limWidth, encoding=self.encoding)
                        if hsw:
                            words[0:0] = hsw
                            self._hyphenations += 1
                            forcedSplit = 1
                            continue
                        elif len(cLine):
                            nMW = maxWidths[min(maxlineno, lineno)]
                            if hyphenation2 or word._fsww + hyw + 1e-08 <= nMW:
                                hsw = word.__shysplit__(fontName, fontSize, 0 + hyw - 1e-08, nMW, encoding=self.encoding)
                                if hsw:
                                    words[0:0] = [word]
                                    forcedSplit = 1
                                    word = None
                                    newWidth = currentWidth
                    elif attemptHyphenation:
                        hyOk = not getattr(f, 'nobr', False)
                        hsw = _hyphenateWord(hyphenator if hyOk else None, fontName, fontSize, word, wordWidth, newWidth, limWidth, uriWasteReduce if hyOk else False, embeddedHyphenation and hyOk, hymwl)
                        if hsw:
                            words[0:0] = hsw
                            self._hyphenations += 1
                            forcedSplit = 1
                            continue
                        elif hyphenation2 and len(cLine):
                            hsw = _hyphenateWord(hyphenator if hyOk else None, fontName, fontSize, word, wordWidth, wordWidth, maxWidth, uriWasteReduce if hyOk else False, embeddedHyphenation and hyOk, hymwl)
                            if hsw:
                                words[0:0] = [word]
                                forcedSplit = 1
                                newWidth = currentWidth
                                word = None
                    if splitLongWords and (not (isinstance(word, _SplitWord) or forcedSplit)):
                        nmw = min(lineno, maxlineno)
                        if wordWidth > max(maxWidths[nmw:nmw + 1]):
                            words[0:0] = _splitWord(word, currentWidth + spaceWidth, maxWidths, lineno, fontName, fontSize, self.encoding)
                            self._splitLongWordCount += 1
                            forcedSplit = 1
                            continue
                if newWidth <= limWidth or not len(cLine) or forcedSplit:
                    if word:
                        cLine.append(word)
                    if forcedSplit:
                        forcedSplit = 0
                        if newWidth > self._width_max:
                            self._width_max = newWidth
                        lines.append((maxWidth - newWidth, cLine))
                        cLine = []
                        currentWidth = -spaceWidth
                        lineno += 1
                        maxWidth = maxWidths[min(maxlineno, lineno)]
                    else:
                        currentWidth = newWidth
                else:
                    if currentWidth > self._width_max:
                        self._width_max = currentWidth
                    lines.append((maxWidth - currentWidth, cLine))
                    cLine = [word]
                    currentWidth = wordWidth
                    lineno += 1
                    maxWidth = maxWidths[min(maxlineno, lineno)]
            if cLine != []:
                if currentWidth > self._width_max:
                    self._width_max = currentWidth
                lines.append((maxWidth - currentWidth, cLine))
            return f.clone(kind=0, lines=lines, ascent=ascent, descent=descent, fontSize=fontSize)
        elif nFrags <= 0:
            return ParaLines(kind=0, fontSize=style.fontSize, fontName=style.fontName, textColor=style.textColor, ascent=style.fontSize, descent=-0.2 * style.fontSize, lines=[])
        else:
            njlbv = not style.justifyBreaks
            words = []
            FW = []
            aFW = FW.append
            _words = _getFragWords(frags, maxWidth)
            sFW = 0
            while _words:
                w = _words.pop(0)
                aFW(w)
                f = w[-1][0]
                fontName = f.fontName
                fontSize = f.fontSize
                if not words:
                    n = spaceWidth = currentWidth = 0
                    maxSize = fontSize
                    maxAscent, minDescent = getAscentDescent(fontName, fontSize)
                wordWidth = w[0]
                f = w[1][0]
                if wordWidth > 0:
                    newWidth = currentWidth + spaceWidth + wordWidth
                else:
                    newWidth = currentWidth
                lineBreak = f._fkind == _FK_BREAK
                limWidth = maxWidth
                if spaceShrinkage:
                    spaceShrink = spaceWidth
                    for wi in words:
                        if wi._fkind == _FK_TEXT:
                            ns = wi.text.count(' ')
                            if ns:
                                spaceShrink += ns * stringWidth(' ', wi.fontName, wi.fontSize)
                    spaceShrink *= spaceShrinkage
                    limWidth += spaceShrink
                if not lineBreak and newWidth > limWidth and (not isinstance(w, _SplitFragH)) and (not hasattr(f, 'cbDefn')):
                    if isinstance(w, _SHYWord):
                        hsw = w.shyphenate(newWidth, limWidth)
                        if hsw:
                            _words[0:0] = hsw
                            _words.insert(1, _InjectedFrag([0, (f.clone(_fkind=_FK_BREAK, text=''), '')]))
                            FW.pop(-1)
                            self._hyphenations += 1
                            continue
                        elif len(FW) > 1:
                            nMW = maxWidths[min(maxlineno, lineno)]
                            if hyphenation2 or w._fsww + 1e-08 <= nMW:
                                hsw = w.shyphenate(wordWidth, nMW)
                                if hsw:
                                    _words[0:0] = [_InjectedFrag([0, (f.clone(_fkind=_FK_BREAK, text=''), '')]), w]
                                    FW.pop(-1)
                                    continue
                    elif attemptHyphenation:
                        hyOk = not getattr(f, 'nobr', False)
                        hsw = _hyphenateFragWord(hyphenator if hyOk else None, w, newWidth, limWidth, uriWasteReduce if hyOk else False, embeddedHyphenation and hyOk, hymwl)
                        if hsw:
                            _words[0:0] = hsw
                            _words.insert(1, _InjectedFrag([0, (f.clone(_fkind=_FK_BREAK, text=''), '')]))
                            FW.pop(-1)
                            self._hyphenations += 1
                            continue
                        elif hyphenation2 and len(FW) > 1:
                            hsw = _hyphenateFragWord(hyphenator if hyOk else None, w, wordWidth, maxWidth, uriWasteReduce if hyOk else False, embeddedHyphenation and hyOk, hymwl)
                            if hsw:
                                _words[0:0] = [_InjectedFrag([0, (f.clone(_fkind=_FK_BREAK, text=''), '')]), w]
                                FW.pop(-1)
                                continue
                    if splitLongWords and (not isinstance(w, _SplitFrag)):
                        nmw = min(lineno, maxlineno)
                        if wordWidth > max(maxWidths[nmw:nmw + 1]):
                            _words[0:0] = _splitFragWord(w, maxWidth - spaceWidth - currentWidth, maxWidths, lineno)
                            _words.insert(1, _InjectedFrag([0, (f.clone(_fkind=_FK_BREAK, text=''), '')]))
                            FW.pop(-1)
                            self._splitLongWordCount += 1
                            continue
                endLine = newWidth > limWidth and n > 0 or lineBreak
                if not endLine:
                    if lineBreak:
                        continue
                    nText = w[1][1]
                    if nText:
                        n += 1
                    fontSize = f.fontSize
                    if calcBounds:
                        if f._fkind == _FK_IMG:
                            descent, ascent = imgVRange(imgNormV(f.cbDefn.height, fontSize), f.cbDefn.valign, fontSize)
                        else:
                            ascent, descent = getAscentDescent(f.fontName, fontSize)
                    else:
                        ascent, descent = getAscentDescent(f.fontName, fontSize)
                    maxSize = max(maxSize, fontSize)
                    maxAscent = max(maxAscent, ascent)
                    minDescent = min(minDescent, descent)
                    if not words:
                        g = f.clone()
                        words = [g]
                        g.text = nText
                    elif not sameFrag(g, f):
                        if spaceWidth:
                            i = len(words) - 1
                            while i >= 0:
                                wi = words[i]
                                i -= 1
                                if wi._fkind == _FK_TEXT:
                                    if not wi.text.endswith(' '):
                                        wi.text += ' '
                                    break
                        g = f.clone()
                        words.append(g)
                        g.text = nText
                    elif spaceWidth:
                        if not g.text.endswith(' '):
                            g.text += ' ' + nText
                        else:
                            g.text += nText
                    else:
                        g.text += nText
                    spaceWidth = stringWidth(' ', fontName, fontSize) if isinstance(w, _HSFrag) else 0
                    ni = 0
                    for i in w[2:]:
                        g = i[0].clone()
                        g.text = i[1]
                        if g.text:
                            ni = 1
                        words.append(g)
                        fontSize = g.fontSize
                        if calcBounds:
                            if g._fkind == _FK_IMG:
                                descent, ascent = imgVRange(imgNormV(g.cbDefn.height, fontSize), g.cbDefn.valign, fontSize)
                            else:
                                ascent, descent = getAscentDescent(g.fontName, fontSize)
                        else:
                            ascent, descent = getAscentDescent(g.fontName, fontSize)
                        maxSize = max(maxSize, fontSize)
                        maxAscent = max(maxAscent, ascent)
                        minDescent = min(minDescent, descent)
                    if not nText and ni:
                        n += 1
                    currentWidth = newWidth
                else:
                    if lineBreak:
                        g = f.clone()
                        words.append(g)
                        llb = njlbv and (not isinstance(w, _InjectedFrag))
                    else:
                        llb = False
                    if currentWidth > self._width_max:
                        self._width_max = currentWidth
                    lines.append(FragLine(extraSpace=maxWidth - currentWidth, wordCount=n, lineBreak=llb, words=words, fontSize=maxSize, ascent=maxAscent, descent=minDescent, maxWidth=maxWidth, sFW=sFW))
                    sFW = len(FW) - 1
                    lineno += 1
                    maxWidth = maxWidths[min(maxlineno, lineno)]
                    if lineBreak:
                        words = []
                        continue
                    spaceWidth = stringWidth(' ', fontName, fontSize) if isinstance(w, _HSFrag) else 0
                    dSpaceShrink = spaceWidth * spaceShrinkage
                    currentWidth = wordWidth
                    n = 1
                    g = f.clone()
                    maxSize = g.fontSize
                    if calcBounds:
                        if g._fkind == _FK_IMG:
                            descent, ascent = imgVRange(imgNormV(g.cbDefn.height, fontSize), g.cbDefn.valign, fontSize)
                        else:
                            maxAscent, minDescent = getAscentDescent(g.fontName, maxSize)
                    else:
                        maxAscent, minDescent = getAscentDescent(g.fontName, maxSize)
                    words = [g]
                    g.text = w[1][1]
                    for i in w[2:]:
                        g = i[0].clone()
                        g.text = i[1]
                        words.append(g)
                        fontSize = g.fontSize
                        if calcBounds:
                            if g._fkind == _FK_IMG:
                                descent, ascent = imgVRange(imgNormV(g.cbDefn.height, fontSize), g.cbDefn.valign, fontSize)
                            else:
                                ascent, descent = getAscentDescent(g.fontName, fontSize)
                        else:
                            ascent, descent = getAscentDescent(g.fontName, fontSize)
                        maxSize = max(maxSize, fontSize)
                        maxAscent = max(maxAscent, ascent)
                        minDescent = min(minDescent, descent)
            if words:
                if currentWidth > self._width_max:
                    self._width_max = currentWidth
                lines.append(ParaLines(extraSpace=maxWidth - currentWidth, wordCount=n, lineBreak=False, words=words, fontSize=maxSize, ascent=maxAscent, descent=minDescent, maxWidth=maxWidth, sFW=sFW))
            self.frags = FW
            return ParaLines(kind=1, lines=lines)

    def breakLinesCJK(self, maxWidths):
        """Initially, the dumbest possible wrapping algorithm.
        Cannot handle font variations."""
        if not isinstance(maxWidths, (list, tuple)):
            maxWidths = [maxWidths]
        style = self.style
        self.height = 0
        _handleBulletWidth(self.bulletText, style, maxWidths)
        frags = self.frags
        nFrags = len(frags)
        if nFrags == 1 and (not hasattr(frags[0], 'cbDefn')) and (not style.endDots):
            f = frags[0]
            if hasattr(self, 'blPara') and getattr(self, '_splitpara', 0):
                return f.clone(kind=0, lines=self.blPara.lines)
            lines = []
            lineno = 0
            if hasattr(f, 'text'):
                text = f.text
            else:
                text = ''.join(getattr(f, 'words', []))
            lines = wordSplit(text, maxWidths, f.fontName, f.fontSize)
            wrappedLines = [(sp, [line]) for sp, line in lines]
            return f.clone(kind=0, lines=wrappedLines, ascent=f.fontSize, descent=-0.2 * f.fontSize)
        elif nFrags <= 0:
            return ParaLines(kind=0, fontSize=style.fontSize, fontName=style.fontName, textColor=style.textColor, lines=[], ascent=style.fontSize, descent=-0.2 * style.fontSize)
        if hasattr(self, 'blPara') and getattr(self, '_splitpara', 0):
            return self.blPara
        autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
        calcBounds = autoLeading not in ('', 'off')
        return cjkFragSplit(frags, maxWidths, calcBounds)

    def beginText(self, x, y):
        return self.canv.beginText(x, y)

    def drawPara(self, debug=0):
        """Draws a paragraph according to the given style.
        Returns the final y position at the bottom. Not safe for
        paragraphs without spaces e.g. Japanese; wrapping
        algorithm will go infinite."""
        canvas = self.canv
        style = self.style
        blPara = self.blPara
        lines = blPara.lines
        leading = style.leading
        autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
        leftIndent = style.leftIndent
        cur_x = leftIndent
        if debug:
            bw = 0.5
            bc = Color(1, 1, 0)
            bg = Color(0.9, 0.9, 0.9)
        else:
            bw = getattr(style, 'borderWidth', None)
            bc = getattr(style, 'borderColor', None)
            bg = style.backColor
        if bg or (bc and bw):
            canvas.saveState()
            op = canvas.rect
            kwds = dict(fill=0, stroke=0)
            if bc and bw:
                canvas.setStrokeColor(bc)
                canvas.setLineWidth(bw)
                kwds['stroke'] = 1
                br = getattr(style, 'borderRadius', 0)
                if br and (not debug):
                    op = canvas.roundRect
                    kwds['radius'] = br
            if bg:
                canvas.setFillColor(bg)
                kwds['fill'] = 1
            bp = getattr(style, 'borderPadding', 0)
            tbp, rbp, bbp, lbp = normalizeTRBL(bp)
            op(leftIndent - lbp, -bbp, self.width - (leftIndent + style.rightIndent) + lbp + rbp, self.height + tbp + bbp, **kwds)
            canvas.restoreState()
        nLines = len(lines)
        bulletText = self.bulletText
        if nLines > 0:
            _offsets = getattr(self, '_offsets', [0])
            _offsets += (nLines - len(_offsets)) * [_offsets[-1]]
            canvas.saveState()
            alignment = style.alignment
            offset = style.firstLineIndent + _offsets[0]
            lim = nLines - 1
            noJustifyLast = not getattr(self, '_JustifyLast', False)
            jllwc = style.justifyLastLine
            isRTL = style.wordWrap == 'RTL'
            bRTL = isRTL and self._wrapWidths or False
            if blPara.kind == 0:
                if alignment == TA_LEFT:
                    dpl = _leftDrawParaLine
                elif alignment == TA_CENTER:
                    dpl = _centerDrawParaLine
                elif alignment == TA_RIGHT:
                    dpl = _rightDrawParaLine
                elif alignment == TA_JUSTIFY:
                    dpl = _justifyDrawParaLineRTL if isRTL else _justifyDrawParaLine
                f = blPara
                if paraFontSizeHeightOffset:
                    cur_y = self.height - f.fontSize
                else:
                    cur_y = self.height - getattr(f, 'ascent', f.fontSize)
                if bulletText:
                    offset = _drawBullet(canvas, offset, cur_y, bulletText, style, rtl=bRTL)
                canvas.setFillColor(f.textColor)
                tx = self.beginText(cur_x, cur_y)
                tx.preformatted = 'preformatted' in self.__class__.__name__.lower()
                if autoLeading == 'max':
                    leading = max(leading, blPara.ascent - blPara.descent)
                elif autoLeading == 'min':
                    leading = blPara.ascent - blPara.descent
                tx.direction = self.style.wordWrap
                tx.setFont(f.fontName, f.fontSize, leading)
                ws = lines[0][0]
                words = lines[0][1]
                lastLine = noJustifyLast and nLines == 1
                if lastLine and jllwc and (len(words) > jllwc):
                    lastLine = False
                t_off = dpl(tx, offset, ws, words, lastLine)
                if f.us_lines or f.link:
                    tx._do_line = MethodType(_do_line, tx)
                    tx.xs = xs = tx.XtraState = ABag()
                    _setTXLineProps(tx, canvas, style)
                    xs.cur_y = cur_y
                    xs.f = f
                    xs.style = style
                    xs.lines = lines
                    xs.link = f.link
                    xs.textColor = f.textColor
                    xs.backColors = []
                    dx = t_off + leftIndent
                    if alignment != TA_JUSTIFY or lastLine:
                        ws = 0
                    if f.us_lines:
                        _do_under_line(0, dx, ws, tx, f.us_lines)
                    if f.link:
                        _do_link_line(0, dx, ws, tx)
                    for i in range(1, nLines):
                        ws = lines[i][0]
                        words = lines[i][1]
                        lastLine = noJustifyLast and i == lim
                        if lastLine and jllwc and (len(words) > jllwc):
                            lastLine = False
                        t_off = dpl(tx, _offsets[i], ws, words, lastLine)
                        dx = t_off + leftIndent
                        if alignment != TA_JUSTIFY or lastLine:
                            ws = 0
                        if f.us_lines:
                            _do_under_line(i, t_off, ws, tx, f.us_lines)
                        if f.link:
                            _do_link_line(i, dx, ws, tx)
                else:
                    for i in range(1, nLines):
                        words = lines[i][1]
                        lastLine = noJustifyLast and i == lim
                        if lastLine and jllwc and (len(words) > jllwc):
                            lastLine = False
                        dpl(tx, _offsets[i], lines[i][0], words, lastLine)
            else:
                if isRTL:
                    for line in lines:
                        line.words = line.words[::-1]
                f = lines[0]
                if paraFontSizeHeightOffset:
                    cur_y = self.height - f.fontSize
                else:
                    cur_y = self.height - getattr(f, 'ascent', f.fontSize)
                dpl = _leftDrawParaLineX
                if bulletText:
                    oo = offset
                    offset = _drawBullet(canvas, offset, cur_y, bulletText, style, rtl=bRTL)
                if alignment == TA_LEFT:
                    dpl = _leftDrawParaLineX
                elif alignment == TA_CENTER:
                    dpl = _centerDrawParaLineX
                elif alignment == TA_RIGHT:
                    dpl = _rightDrawParaLineX
                elif alignment == TA_JUSTIFY:
                    dpl = _justifyDrawParaLineXRTL if isRTL else _justifyDrawParaLineX
                else:
                    raise ValueError('bad align %s' % repr(alignment))
                tx = self.beginText(cur_x, cur_y)
                tx.preformatted = 'preformatted' in self.__class__.__name__.lower()
                _setTXLineProps(tx, canvas, style)
                tx._do_line = MethodType(_do_line, tx)
                tx.direction = self.style.wordWrap
                xs = tx.XtraState = ABag()
                xs.textColor = None
                xs.backColor = None
                xs.rise = 0
                xs.backColors = []
                xs.us_lines = {}
                xs.links = {}
                xs.link = {}
                xs.leading = style.leading
                xs.leftIndent = leftIndent
                tx._leading = None
                tx._olb = None
                xs.cur_y = cur_y
                xs.f = f
                xs.style = style
                xs.autoLeading = autoLeading
                xs.paraWidth = self.width
                tx._fontname, tx._fontsize = (None, None)
                line = lines[0]
                lastLine = noJustifyLast and nLines == 1
                if lastLine and jllwc and (line.wordCount > jllwc):
                    lastLine = False
                dpl(tx, offset, line, lastLine)
                _do_post_text(tx)
                for i in range(1, nLines):
                    line = lines[i]
                    lastLine = noJustifyLast and i == lim
                    if lastLine and jllwc and (line.wordCount > jllwc):
                        lastLine = False
                    dpl(tx, _offsets[i], line, lastLine)
                    _do_post_text(tx)
            canvas.drawText(tx)
            canvas.restoreState()

    def getPlainText(self, identify=None):
        """Convenience function for templates which want access
        to the raw text, without XML tags. """
        frags = getattr(self, 'frags', None)
        if frags:
            plains = []
            plains_append = plains.append
            if _processed_frags(frags):
                for word in frags:
                    for style, text in word[1:]:
                        plains_append(text)
                    if isinstance(word, _HSFrag):
                        plains_append(' ')
            else:
                for frag in frags:
                    if hasattr(frag, 'text'):
                        plains_append(frag.text)
            return ''.join(plains)
        elif identify:
            text = getattr(self, 'text', None)
            if text is None:
                text = repr(self)
            return text
        else:
            return ''

    def getActualLineWidths0(self):
        """Convenience function; tells you how wide each line
        actually is.  For justified styles, this will be
        the same as the wrap width; for others it might be
        useful for seeing if paragraphs will fit in spaces."""
        assert hasattr(self, 'width'), 'Cannot call this method before wrap()'
        if self.blPara.kind:
            func = lambda frag, w=self.width: w - frag.extraSpace
        else:
            func = lambda frag, w=self.width: w - frag[0]
        return list(map(func, self.blPara.lines))

    @staticmethod
    def dumpFrags(frags, indent=4, full=False):
        R = ['[']
        aR = R.append
        for i, f in enumerate(frags):
            if full:
                aR('    [%r,' % f[0])
                for fx in f[1:]:
                    aR('        (%s,)' % repr(fx[0]))
                    aR('        %r),' % fx[1])
                    aR('    ], #%d %s' % (i, f.__class__.__name__))
                aR('    ]')
            else:
                aR('[%r, %s], #%d %s' % (f[0], ', '.join(('(%s,%r)' % (fx[0].__class__.__name__, fx[1]) for fx in f[1:])), i, f.__class__.__name__))
        i = indent * ' '
        return i + ('\n' + i).join(R)
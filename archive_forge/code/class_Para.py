from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class Para(Flowable):
    spaceBefore = 0
    spaceAfter = 0

    def __init__(self, style, parsedText=None, bulletText=None, state=None, context=None, baseindent=0):
        self.baseindent = baseindent
        self.context = buildContext(context)
        self.parsedText = parsedText
        self.bulletText = bulletText
        self.style1 = style
        self.program = []
        self.formattedProgram = []
        self.remainder = None
        self.state = state
        if not state:
            self.spaceBefore = style.spaceBefore
            self.spaceAfter = style.spaceAfter
        self.bold = 0
        self.italic = 0
        self.face = style.fontName
        self.size = style.fontSize

    def getSpaceBefore(self):
        return self.spaceBefore

    def getSpaceAfter(self):
        return self.spaceAfter

    def wrap(self, availableWidth, availableHeight):
        if debug:
            print('WRAPPING', id(self), availableWidth, availableHeight)
            print('   ', self.formattedProgram)
            print('   ', self.program)
        self.availableHeight = availableHeight
        self.myengine = p = paragraphEngine()
        p.baseindent = self.baseindent
        parsedText = self.parsedText
        formattedProgram = self.formattedProgram
        state = self.state
        if state:
            leading = state['leading']
        else:
            leading = self.style1.leading
        program = self.program
        self.cansplit = 1
        if state:
            p.resetState(state)
            p.x = 0
            p.y = 0
            needatleast = state['leading']
        else:
            needatleast = self.style1.leading
        if availableHeight <= needatleast:
            self.cansplit = 0
            return (availableHeight + 1, availableWidth)
        if parsedText is None and program is None:
            raise ValueError('need parsedText for formatting')
        if not program:
            self.program = program = self.compileProgram(parsedText)
        if not self.formattedProgram:
            formattedProgram, remainder, laststate, heightused = p.format(availableWidth, availableHeight, program, leading)
            self.formattedProgram = formattedProgram
            self.height = heightused
            self.laststate = laststate
            self.remainderProgram = remainder
        else:
            heightused = self.height
            remainder = None
        if remainder:
            height = availableHeight + 1
            self.remainder = Para(self.style1, parsedText=None, bulletText=None, state=laststate, context=self.context)
            self.remainder.program = remainder
            self.remainder.spaceAfter = self.spaceAfter
            self.spaceAfter = 0
        else:
            self.remainder = None
            height = heightused
            if height > availableHeight:
                height = availableHeight - 0.1
        result = (availableWidth, height)
        if debug:
            w, h = result
            if abs(availableHeight - h) < 0.2:
                print('exact match???' + repr(availableHeight, h))
            print('wrap is', (availableWidth, availableHeight), result)
        return result

    def split(self, availableWidth, availableHeight):
        if availableHeight <= 0 or not self.cansplit:
            return []
        self.availableHeight = availableHeight
        formattedProgram = self.formattedProgram
        if formattedProgram is None:
            raise ValueError('must call wrap before split')
        elif not formattedProgram:
            return []
        remainder = self.remainder
        if remainder:
            result = [self, remainder]
        else:
            result = [self]
        return result

    def draw(self):
        p = self.myengine
        formattedProgram = self.formattedProgram
        if formattedProgram is None:
            raise ValueError('must call wrap before draw')
        state = self.state
        laststate = self.laststate
        if state:
            p.resetState(state)
            p.x = 0
            p.y = 0
        c = self.canv
        height = self.height
        if state:
            leading = state['leading']
        else:
            leading = self.style1.leading
        c.translate(0, height - self.size)
        t = c.beginText()
        if DUMPPROGRAM or debug:
            print('=' * 44, 'now running program')
            for x in formattedProgram:
                print(x)
            print('-' * 44)
        laststate = p.runOpCodes(formattedProgram, c, t)
        c.drawText(t)

    def compileProgram(self, parsedText, program=None):
        style = self.style1
        if program is None:
            program = []
        a = program.append
        fn = style.fontName
        a(('face', fn))
        from reportlab.lib.fonts import ps2tt
        self.face, self.bold, self.italic = ps2tt(fn)
        a(('size', style.fontSize))
        self.size = style.fontSize
        a(('align', style.alignment))
        a(('indent', style.leftIndent))
        if style.firstLineIndent:
            a(('indent', style.firstLineIndent))
        a(('rightIndent', style.rightIndent))
        a(('leading', style.leading))
        if style.textColor:
            a(('color', style.textColor))
        if self.bulletText:
            self.do_bullet(self.bulletText, program)
        self.compileComponent(parsedText, program)
        if style.firstLineIndent:
            count = 0
            for x in program:
                count += 1
                if isinstance(x, str) or hasattr(x, 'width'):
                    break
            program.insert(count, ('indent', -style.firstLineIndent))
        return program

    def linearize(self, program=None, parsedText=None):
        if parsedText is None:
            parsedText = self.parsedText
        style = self.style1
        if program is None:
            program = []
        program.append(('push',))
        if style.spaceBefore:
            program.append(('leading', style.spaceBefore + style.leading))
        else:
            program.append(('leading', style.leading))
        program.append(('nextLine', 0))
        program = self.compileProgram(parsedText, program=program)
        program.append(('pop',))
        program.append(('push',))
        if style.spaceAfter:
            program.append(('leading', style.spaceAfter))
        else:
            program.append(('leading', 0))
        program.append(('nextLine', 0))
        program.append(('pop',))

    def compileComponent(self, parsedText, program):
        if isinstance(parsedText, str):
            if parsedText:
                stext = parsedText.strip()
                if not stext:
                    program.append(' ')
                else:
                    handleSpecialCharacters(self, parsedText, program)
        elif isinstance(parsedText, list):
            for e in parsedText:
                self.compileComponent(e, program)
        elif isinstance(parsedText, tuple):
            tagname, attdict, content, extra = parsedText
            if not attdict:
                attdict = {}
            compilername = 'compile_' + tagname
            compiler = getattr(self, compilername, None)
            if compiler is not None:
                compiler(attdict, content, extra, program)
            elif debug:
                L = ['<' + tagname]
                a = L.append
                if not attdict:
                    attdict = {}
                for k, v in attdict.items():
                    a(' %s=%s' % (k, v))
                if content:
                    a('>')
                    a(str(content))
                    a('</%s>' % tagname)
                else:
                    a('/>')
                t = ''.join(L)
                handleSpecialCharacters(self, t, program)
            else:
                raise ValueError("don't know how to handle tag " + repr(tagname))

    def shiftfont(self, program, face=None, bold=None, italic=None):
        oldface = self.face
        oldbold = self.bold
        olditalic = self.italic
        oldfontinfo = (oldface, oldbold, olditalic)
        if face is None:
            face = oldface
        if bold is None:
            bold = oldbold
        if italic is None:
            italic = olditalic
        self.face = face
        self.bold = bold
        self.italic = italic
        from reportlab.lib.fonts import tt2ps
        font = tt2ps(face, bold, italic)
        oldfont = tt2ps(oldface, oldbold, olditalic)
        if font != oldfont:
            program.append(('face', font))
        return oldfontinfo

    def compile_(self, attdict, content, extra, program):
        for e in content:
            self.compileComponent(e, program)

    def compile_pageNumber(self, attdict, content, extra, program):
        program.append(PageNumberObject())

    def compile_b(self, attdict, content, extra, program):
        f, b, i = self.shiftfont(program, bold=1)
        for e in content:
            self.compileComponent(e, program)
        self.shiftfont(program, bold=b)

    def compile_i(self, attdict, content, extra, program):
        f, b, i = self.shiftfont(program, italic=1)
        for e in content:
            self.compileComponent(e, program)
        self.shiftfont(program, italic=i)

    def compile_u(self, attdict, content, extra, program):
        program.append(('lineOperation', UNDERLINE))
        for e in content:
            self.compileComponent(e, program)
        program.append(('endLineOperation', UNDERLINE))

    def compile_sub(self, attdict, content, extra, program):
        size = self.size
        self.size = newsize = size * 0.7
        rise = size * 0.5
        program.append(('size', newsize))
        self.size = size
        program.append(('rise', -rise))
        for e in content:
            self.compileComponent(e, program)
        program.append(('size', size))
        program.append(('rise', rise))

    def compile_ul(self, attdict, content, extra, program, tagname='ul'):
        atts = attdict.copy()
        bulletmaker = bulletMaker(tagname, atts, self.context)
        for e in content:
            if isinstance(e, str):
                if e.strip():
                    raise ValueError("don't expect CDATA between list elements")
            elif isinstance(e, tuple):
                tagname, attdict1, content1, extra = e
                if tagname != 'li':
                    raise ValueError("don't expect %s inside list" % repr(tagname))
                newatts = atts.copy()
                if attdict1:
                    newatts.update(attdict1)
                bulletmaker.makeBullet(newatts)
                self.compile_para(newatts, content1, extra, program)

    def compile_ol(self, attdict, content, extra, program):
        return self.compile_ul(attdict, content, extra, program, tagname='ol')

    def compile_dl(self, attdict, content, extra, program):
        atts = attdict.copy()
        atts = attdict.copy()
        bulletmaker = bulletMaker('dl', atts, self.context)
        contentcopy = list(content)
        bullet = ''
        while contentcopy:
            e = contentcopy[0]
            del contentcopy[0]
            if isinstance(e, str):
                if e.strip():
                    raise ValueError("don't expect CDATA between list elements")
                elif not contentcopy:
                    break
                else:
                    continue
            elif isinstance(e, tuple):
                tagname, attdict1, content1, extra = e
                if tagname != 'dd' and tagname != 'dt':
                    raise ValueError("don't expect %s here inside list, expect 'dd' or 'dt'" % repr(tagname))
                if tagname == 'dt':
                    if bullet:
                        raise ValueError('dt will not be displayed unless followed by a dd: ' + repr(bullet))
                    if content1:
                        self.compile_para(attdict1, content1, extra, program)
                elif tagname == 'dd':
                    newatts = atts.copy()
                    if attdict1:
                        newatts.update(attdict1)
                    bulletmaker.makeBullet(newatts, bl=bullet)
                    self.compile_para(newatts, content1, extra, program)
                    bullet = ''
        if bullet:
            raise ValueError('dt will not be displayed unless followed by a dd' + repr(bullet))

    def compile_super(self, attdict, content, extra, program):
        size = self.size
        self.size = newsize = size * 0.7
        rise = size * 0.5
        program.append(('size', newsize))
        program.append(('rise', rise))
        for e in content:
            self.compileComponent(e, program)
        program.append(('size', size))
        self.size = size
        program.append(('rise', -rise))

    def compile_font(self, attdict, content, extra, program):
        program.append(('push',))
        if 'face' in attdict:
            face = attdict['face']
            from reportlab.lib.fonts import tt2ps
            try:
                font = tt2ps(face, self.bold, self.italic)
            except:
                font = face
            program.append(('face', font))
        if 'color' in attdict:
            colorname = attdict['color']
            program.append(('color', colorname))
        if 'size' in attdict:
            size = attdict['size']
            program.append(('size', size))
        for e in content:
            self.compileComponent(e, program)
        program.append(('pop',))

    def compile_a(self, attdict, content, extra, program):
        url = attdict['href']
        colorname = attdict.get('color', 'blue')
        Link = HotLink(url)
        program.append(('push',))
        program.append(('color', colorname))
        program.append(('lineOperation', Link))
        program.append(('lineOperation', UNDERLINE))
        for e in content:
            self.compileComponent(e, program)
        program.append(('endLineOperation', UNDERLINE))
        program.append(('endLineOperation', Link))
        program.append(('pop',))

    def compile_link(self, attdict, content, extra, program):
        dest = attdict['destination']
        colorname = attdict.get('color', None)
        Link = InternalLink(dest)
        program.append(('push',))
        if colorname:
            program.append(('color', colorname))
        program.append(('lineOperation', Link))
        program.append(('lineOperation', UNDERLINE))
        for e in content:
            self.compileComponent(e, program)
        program.append(('endLineOperation', UNDERLINE))
        program.append(('endLineOperation', Link))
        program.append(('pop',))

    def compile_setLink(self, attdict, content, extra, program):
        dest = attdict['destination']
        colorname = attdict.get('color', 'blue')
        Link = DefDestination(dest)
        program.append(('push',))
        if colorname:
            program.append(('color', colorname))
        program.append(('lineOperation', Link))
        if colorname:
            program.append(('lineOperation', UNDERLINE))
        for e in content:
            self.compileComponent(e, program)
        if colorname:
            program.append(('endLineOperation', UNDERLINE))
        program.append(('endLineOperation', Link))
        program.append(('pop',))

    def compile_bullet(self, attdict, content, extra, program):
        if len(content) != 1 or not isinstance(content[0], str):
            raise ValueError('content for bullet must be a single string')
        text = content[0]
        self.do_bullet(text, program)

    def do_bullet(self, text, program):
        style = self.style1
        indent = style.bulletIndent + self.baseindent
        font = style.bulletFontName
        size = style.bulletFontSize
        program.append(('bullet', text, indent, font, size))

    def compile_tt(self, attdict, content, extra, program):
        f, b, i = self.shiftfont(program, face='Courier')
        for e in content:
            self.compileComponent(e, program)
        self.shiftfont(program, face=f)

    def compile_greek(self, attdict, content, extra, program):
        self.compile_font({'face': 'symbol'}, content, extra, program)

    def compile_evalString(self, attdict, content, extra, program):
        program.append(EvalStringObject(attdict, content, extra, self.context))

    def compile_name(self, attdict, content, extra, program):
        program.append(NameObject(attdict, content, extra, self.context))

    def compile_getName(self, attdict, content, extra, program):
        program.append(GetNameObject(attdict, content, extra, self.context))

    def compile_seq(self, attdict, content, extra, program):
        program.append(SeqObject(attdict, content, extra, self.context))

    def compile_seqReset(self, attdict, content, extra, program):
        program.append(SeqResetObject(attdict, content, extra, self.context))

    def compile_seqDefault(self, attdict, content, extra, program):
        program.append(SeqDefaultObject(attdict, content, extra, self.context))

    def compile_para(self, attdict, content, extra, program, stylename='para.defaultStyle'):
        if attdict is None:
            attdict = {}
        context = self.context
        stylename = attdict.get('style', stylename)
        style = context[stylename]
        newstyle = SimpleStyle(name='rml2pdf internal embedded style', parent=style)
        newstyle.addAttributes(attdict)
        bulletText = attdict.get('bulletText', None)
        mystyle = self.style1
        thepara = Para(newstyle, content, context=context, bulletText=bulletText)
        mybaseindent = self.baseindent
        self.baseindent = thepara.baseindent = mystyle.leftIndent + self.baseindent
        thepara.linearize(program=program)
        program.append(('nextLine', 0))
        self.baseindent = mybaseindent
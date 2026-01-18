from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class paragraphEngine:

    def __init__(self, program=None):
        from reportlab.lib.colors import black
        if program is None:
            program = []
        self.lineOpHandlers = []
        self.program = program
        self.indent = self.rightIndent = 0.0
        self.baseindent = 0.0
        self.fontName = 'Helvetica'
        self.fontSize = 10
        self.leading = 12
        self.fontColor = black
        self.x = self.y = self.rise = 0.0
        from reportlab.lib.enums import TA_LEFT
        self.alignment = TA_LEFT
        self.textStateStack = []
    TEXT_STATE_VARIABLES = ('indent', 'rightIndent', 'fontName', 'fontSize', 'leading', 'fontColor', 'lineOpHandlers', 'rise', 'alignment')

    def pushTextState(self):
        state = []
        for var in self.TEXT_STATE_VARIABLES:
            val = getattr(self, var)
            state.append(val)
        self.textStateStack = self.textStateStack + [state]
        return state

    def popTextState(self):
        state = self.textStateStack[-1]
        self.textStateStack = self.textStateStack[:-1]
        state = state[:]
        for var in self.TEXT_STATE_VARIABLES:
            val = state[0]
            del state[0]
            setattr(self, var, val)

    def format(self, maxwidth, maxheight, program, leading=0):
        """return program with line operations added if at least one line fits"""
        startstate = self.__dict__.copy()
        remainder = program[:]
        lineprogram = []
        heightremaining = maxheight
        if leading:
            self.leading = leading
        room = 1
        cursorcount = 0
        while remainder and room:
            indent = self.indent
            rightIndent = self.rightIndent
            linewidth = maxwidth - indent - rightIndent
            beforelinestate = self.__dict__.copy()
            if linewidth < TOOSMALLSPACE:
                raise ValueError('indents %s %s too wide for space %s' % (self.indent, self.rightIndent, maxwidth))
            try:
                lineIsFull, line, cursor, currentLength, usedIndent, maxLength, justStrings = self.fitLine(remainder, maxwidth)
            except:
                raise
            cursorcount = cursorcount + cursor
            leading = self.leading
            if heightremaining > leading:
                heightremaining = heightremaining - leading
            else:
                room = 0
                self.__dict__.update(beforelinestate)
                break
            alignment = self.alignment
            remainder = remainder[cursor:]
            if not remainder:
                del line[-1]
            if alignment == TA_LEFT:
                if justStrings:
                    line = stringLine(line, currentLength)
                else:
                    line = self.shrinkWrap(line)
                pass
            elif alignment == TA_CENTER:
                if justStrings:
                    line = stringLine(line, currentLength)
                else:
                    line = self.shrinkWrap(line)
                line = self.centerAlign(line, currentLength, maxLength)
            elif alignment == TA_RIGHT:
                if justStrings:
                    line = stringLine(line, currentLength)
                else:
                    line = self.shrinkWrap(line)
                line = self.rightAlign(line, currentLength, maxLength)
            elif alignment == TA_JUSTIFY:
                if remainder and lineIsFull:
                    if justStrings:
                        line = simpleJustifyAlign(line, currentLength, maxLength)
                    else:
                        line = self.justifyAlign(line, currentLength, maxLength)
                else:
                    if justStrings:
                        line = stringLine(line, currentLength)
                    else:
                        line = self.shrinkWrap(line)
                    if debug:
                        print('no justify because line is not full or end of para')
            else:
                raise ValueError('bad alignment ' + repr(alignment))
            if not justStrings:
                line = self.cleanProgram(line)
            lineprogram.extend(line)
        laststate = self.__dict__.copy()
        self.__dict__.update(startstate)
        heightused = maxheight - heightremaining
        return (lineprogram, remainder, laststate, heightused)

    def getState(self):
        return self.__dict__.copy()

    def resetState(self, state):
        self.__dict__.update(state)

    def fitLine(self, program, totalLength):
        """fit words (and other things) onto a line"""
        from reportlab.pdfbase.pdfmetrics import stringWidth
        usedIndent = self.indent
        maxLength = totalLength - usedIndent - self.rightIndent
        done = 0
        line = []
        cursor = 0
        lineIsFull = 0
        currentLength = 0
        maxcursor = len(program)
        needspace = 0
        first = 1
        terminated = None
        fontName = self.fontName
        fontSize = self.fontSize
        spacewidth = stringWidth(' ', fontName, fontSize)
        justStrings = 1
        while not done and cursor < maxcursor:
            opcode = program[cursor]
            if isinstance(opcode, str) or hasattr(opcode, 'width'):
                lastneedspace = needspace
                needspace = 0
                if hasattr(opcode, 'width'):
                    justStrings = 0
                    width = opcode.width(self)
                    needspace = 0
                else:
                    saveopcode = opcode
                    opcode = opcode.strip()
                    if opcode:
                        width = stringWidth(opcode, fontName, fontSize)
                    else:
                        width = 0
                    if saveopcode and (width or currentLength):
                        needspace = saveopcode[-1] == ' '
                    else:
                        needspace = 0
                fullwidth = width
                if lastneedspace:
                    fullwidth = width + spacewidth
                newlength = currentLength + fullwidth
                if newlength > maxLength and (not first):
                    done = 1
                    lineIsFull = 1
                else:
                    if lastneedspace:
                        line.append(spacewidth)
                    if opcode:
                        line.append(opcode)
                    if abs(width) > TOOSMALLSPACE:
                        line.append(-width)
                        currentLength = newlength
                first = 0
            elif isinstance(opcode, float):
                justStrings = 0
                aopcode = abs(opcode)
                if aopcode > TOOSMALLSPACE:
                    nextLength = currentLength + aopcode
                    if nextLength > maxLength and (not first):
                        done = 1
                    elif aopcode > TOOSMALLSPACE:
                        currentLength = nextLength
                        line.append(opcode)
                    first = 0
            elif isinstance(opcode, tuple):
                justStrings = 0
                indicator = opcode[0]
                if indicator == 'nextLine':
                    line.append(opcode)
                    cursor += 1
                    terminated = done = 1
                elif indicator == 'color':
                    oldcolor = self.fontColor
                    i, colorname = opcode
                    if isinstance(colorname, str):
                        color = self.fontColor = getattr(colors, colorname)
                    else:
                        color = self.fontColor = colorname
                    line.append(opcode)
                elif indicator == 'face':
                    i, fontname = opcode
                    fontName = self.fontName = fontname
                    spacewidth = stringWidth(' ', fontName, fontSize)
                    line.append(opcode)
                elif indicator == 'size':
                    i, fontsize = opcode
                    size = abs(float(fontsize))
                    if isinstance(fontsize, str):
                        if fontsize[:1] == '+':
                            fontSize = self.fontSize = self.fontSize + size
                        elif fontsize[:1] == '-':
                            fontSize = self.fontSize = self.fontSize - size
                        else:
                            fontSize = self.fontSize = size
                    else:
                        fontSize = self.fontSize = size
                    spacewidth = stringWidth(' ', fontName, fontSize)
                    line.append(opcode)
                elif indicator == 'leading':
                    i, leading = opcode
                    self.leading = leading
                    line.append(opcode)
                elif indicator == 'indent':
                    i, increment = opcode
                    indent = self.indent = self.indent + increment
                    if first:
                        usedIndent = max(indent, usedIndent)
                        maxLength = totalLength - usedIndent - self.rightIndent
                    line.append(opcode)
                elif indicator == 'push':
                    self.pushTextState()
                    line.append(opcode)
                elif indicator == 'pop':
                    try:
                        self.popTextState()
                    except:
                        raise
                    fontName = self.fontName
                    fontSize = self.fontSize
                    spacewidth = stringWidth(' ', fontName, fontSize)
                    line.append(opcode)
                elif indicator == 'bullet':
                    i, bullet, indent, font, size = opcode
                    indent = indent + self.baseindent
                    opcode = (i, bullet, indent, font, size)
                    if not first:
                        raise ValueError('bullet not at beginning of line')
                    bulletwidth = float(stringWidth(bullet, font, size))
                    spacewidth = float(stringWidth(' ', font, size))
                    bulletmin = indent + spacewidth + bulletwidth
                    usedIndent = max(bulletmin, usedIndent)
                    if first:
                        maxLength = totalLength - usedIndent - self.rightIndent
                    line.append(opcode)
                elif indicator == 'rightIndent':
                    i, increment = opcode
                    self.rightIndent = self.rightIndent + increment
                    if first:
                        maxLength = totalLength - usedIndent - self.rightIndent
                    line.append(opcode)
                elif indicator == 'rise':
                    i, rise = opcode
                    newrise = self.rise = self.rise + rise
                    line.append(opcode)
                elif indicator == 'align':
                    i, alignment = opcode
                    self.alignment = alignment
                    line.append(opcode)
                elif indicator == 'lineOperation':
                    i, handler = opcode
                    line.append(opcode)
                    self.lineOpHandlers = self.lineOpHandlers + [handler]
                elif indicator == 'endLineOperation':
                    i, handler = opcode
                    h = self.lineOpHandlers[:]
                    h.remove(handler)
                    self.lineOpHandlers = h
                    line.append(opcode)
                else:
                    raise ValueError("at format time don't understand indicator " + repr(indicator))
            else:
                raise ValueError('op must be string, float, instance, or tuple ' + repr(opcode))
            if not done:
                cursor += 1
        if not terminated:
            line.append(('nextLine', 0))
        return (lineIsFull, line, cursor, currentLength, usedIndent, maxLength, justStrings)

    def centerAlign(self, line, lineLength, maxLength):
        diff = maxLength - lineLength
        shift = diff / 2.0
        if shift > TOOSMALLSPACE:
            return self.insertShift(line, shift)
        return line

    def rightAlign(self, line, lineLength, maxLength):
        shift = maxLength - lineLength
        if shift > TOOSMALLSPACE:
            return self.insertShift(line, shift)
        return line

    def insertShift(self, line, shift):
        result = []
        first = 1
        for e in line:
            if first and (isinstance(e, str) or hasattr(e, 'width')):
                result.append(shift)
                first = 0
            result.append(e)
        return result

    def justifyAlign(self, line, lineLength, maxLength):
        diff = maxLength - lineLength
        spacecount = 0
        visible = 0
        first = 1
        for e in line:
            if isinstance(e, float) and e > TOOSMALLSPACE and visible:
                spacecount += 1
            elif first and (isinstance(e, str) or hasattr(e, 'width')):
                visible = 1
                first = 0
        if spacecount < 1:
            return line
        shift = diff / float(spacecount)
        if shift <= TOOSMALLSPACE:
            return line
        first = 1
        visible = 0
        result = []
        cursor = 0
        nline = len(line)
        while cursor < nline:
            e = line[cursor]
            result.append(e)
            if first and (isinstance(e, str) or hasattr(e, 'width')):
                visible = 1
            elif isinstance(e, float) and e > TOOSMALLSPACE and visible:
                expanded = e + shift
                result[-1] = expanded
            cursor += 1
        return result

    def shrinkWrap(self, line):
        result = []
        index = 0
        maxindex = len(line)
        while index < maxindex:
            e = line[index]
            if isinstance(e, str) and index < maxindex - 1:
                thestrings = [e]
                thefloats = 0.0
                index += 1
                nexte = line[index]
                while index < maxindex and isinstance(nexte, (float, str)):
                    if isinstance(nexte, float):
                        if thefloats < 0 and nexte > 0:
                            thefloats = -thefloats
                        if nexte < 0 and thefloats > 0:
                            nexte = -nexte
                        thefloats += nexte
                    elif isinstance(nexte, str):
                        thestrings.append(nexte)
                    index += 1
                    if index < maxindex:
                        nexte = line[index]
                s = ' '.join(thestrings)
                result.append(s)
                result.append(float(thefloats))
                index -= 1
            else:
                result.append(e)
            index += 1
        return result

    def cleanProgram(self, line):
        """collapse adjacent spacings"""
        result = []
        last = 0
        for e in line:
            if isinstance(e, float):
                if last < 0 and e > 0:
                    last = -last
                if e < 0 and last > 0:
                    e = -e
                last = float(last) + e
            else:
                if abs(last) > TOOSMALLSPACE:
                    result.append(last)
                result.append(e)
                last = 0
        if last:
            result.append(last)
        change = 1
        rline = list(range(len(result) - 1))
        while change:
            change = 0
            for index in rline:
                nextindex = index + 1
                this = result[index]
                next = result[nextindex]
                doswap = 0
                if isinstance(this, str) or isinstance(next, str) or hasattr(this, 'width') or hasattr(next, 'width'):
                    doswap = 0
                elif isinstance(this, tuple):
                    thisindicator = this[0]
                    if isinstance(next, tuple):
                        nextindicator = next[0]
                        doswap = 0
                        if nextindicator == 'endLineOperation' and thisindicator != 'endLineOperation' and (thisindicator != 'lineOperation'):
                            doswap = 1
                    elif isinstance(next, float):
                        if thisindicator == 'lineOperation':
                            doswap = 1
                if doswap:
                    result[index] = next
                    result[nextindex] = this
                    change = 1
        return result

    def runOpCodes(self, program, canvas, textobject):
        """render the line(s)"""
        escape = canvas._escape
        code = textobject._code
        startstate = self.__dict__.copy()
        font = None
        size = None
        textobject.setFillColor(self.fontColor)
        xstart = self.x
        thislineindent = self.indent
        thislinerightIndent = self.rightIndent
        indented = 0
        for opcode in program:
            if isinstance(opcode, str) or hasattr(opcode, 'width'):
                if not indented:
                    if abs(thislineindent) > TOOSMALLSPACE:
                        code.append('%s Td' % fp_str(thislineindent, 0))
                        self.x += thislineindent
                    for handler in self.lineOpHandlers:
                        handler.start_at(self.x, self.y, self, canvas, textobject)
                indented = 1
                if font != self.fontName or size != self.fontSize:
                    font = self.fontName
                    size = self.fontSize
                    textobject.setFont(font, size)
                if isinstance(opcode, str):
                    textobject.textOut(opcode)
                else:
                    opcode.execute(self, textobject, canvas)
            elif isinstance(opcode, float):
                opcode = abs(opcode)
                if opcode > TOOSMALLSPACE:
                    code.append('%s Td' % fp_str(opcode, 0))
                    self.x += opcode
            elif isinstance(opcode, tuple):
                indicator = opcode[0]
                if indicator == 'nextLine':
                    i, endallmarks = opcode
                    x = self.x
                    y = self.y
                    newy = self.y = self.y - self.leading
                    newx = self.x = xstart
                    thislineindent = self.indent
                    thislinerightIndent = self.rightIndent
                    indented = 0
                    for handler in self.lineOpHandlers:
                        handler.end_at(x, y, self, canvas, textobject)
                    textobject.setTextOrigin(newx, newy)
                elif indicator == 'color':
                    oldcolor = self.fontColor
                    i, colorname = opcode
                    if isinstance(colorname, str):
                        color = self.fontColor = getattr(colors, colorname)
                    else:
                        color = self.fontColor = colorname
                    if color != oldcolor:
                        textobject.setFillColor(color)
                elif indicator == 'face':
                    i, fontname = opcode
                    self.fontName = fontname
                elif indicator == 'size':
                    i, fontsize = opcode
                    size = abs(float(fontsize))
                    if isinstance(fontsize, str):
                        if fontsize[:1] == '+':
                            self.fontSize += size
                        elif fontsize[:1] == '-':
                            self.fontSize -= size
                        else:
                            self.fontSize = size
                    else:
                        self.fontSize = size
                    fontSize = self.fontSize
                    textobject.setFont(self.fontName, self.fontSize)
                elif indicator == 'leading':
                    i, leading = opcode
                    self.leading = leading
                elif indicator == 'indent':
                    i, increment = opcode
                    indent = self.indent = self.indent + increment
                    thislineindent = max(thislineindent, indent)
                elif indicator == 'push':
                    self.pushTextState()
                elif indicator == 'pop':
                    oldcolor = self.fontColor
                    oldfont = self.fontName
                    oldsize = self.fontSize
                    self.popTextState()
                    if oldcolor != self.fontColor:
                        textobject.setFillColor(self.fontColor)
                elif indicator == 'wordSpacing':
                    i, ws = opcode
                    textobject.setWordSpace(ws)
                elif indicator == 'bullet':
                    i, bullet, indent, font, size = opcode
                    if abs(self.x - xstart) > TOOSMALLSPACE:
                        raise ValueError('bullet not at beginning of line')
                    bulletwidth = float(stringWidth(bullet, font, size))
                    spacewidth = float(stringWidth(' ', font, size))
                    bulletmin = indent + spacewidth + bulletwidth
                    if bulletmin > thislineindent:
                        thislineindent = bulletmin
                    textobject.moveCursor(indent, 0)
                    textobject.setFont(font, size)
                    textobject.textOut(bullet)
                    textobject.moveCursor(-indent, 0)
                    textobject.setFont(self.fontName, self.fontSize)
                elif indicator == 'rightIndent':
                    i, increment = opcode
                    self.rightIndent = self.rightIndent + increment
                elif indicator == 'rise':
                    i, rise = opcode
                    newrise = self.rise = self.rise + rise
                    textobject.setRise(newrise)
                elif indicator == 'align':
                    i, alignment = opcode
                    self.alignment = alignment
                elif indicator == 'lineOperation':
                    i, handler = opcode
                    handler.start_at(self.x, self.y, self, canvas, textobject)
                    self.lineOpHandlers = self.lineOpHandlers + [handler]
                elif indicator == 'endLineOperation':
                    i, handler = opcode
                    handler.end_at(self.x, self.y, self, canvas, textobject)
                    newh = self.lineOpHandlers = self.lineOpHandlers[:]
                    if handler in newh:
                        self.lineOpHandlers.remove(handler)
                    else:
                        pass
                else:
                    raise ValueError("don't understand indicator " + repr(indicator))
            else:
                raise ValueError('op must be string float or tuple ' + repr(opcode))
        laststate = self.__dict__.copy()
        self.__dict__.update(startstate)
        return laststate
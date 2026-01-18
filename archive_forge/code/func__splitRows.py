from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def _splitRows(self, availHeight, doInRowSplit=0):
    n = self._getFirstPossibleSplitRowPosition(availHeight, ignoreSpans=doInRowSplit)
    repeatRows = self.repeatRows
    maxrepeat = repeatRows if isinstance(repeatRows, int) else max(repeatRows) + 1
    if doInRowSplit and n < maxrepeat or (not doInRowSplit and n <= maxrepeat):
        return []
    lim = len(self._rowHeights)
    if n == lim:
        return [self]
    lo = self._rowSplitRange
    if lo:
        lo, hi = lo
        if lo < 0:
            lo += lim
        if hi < 0:
            hi += lim
        if n > hi:
            return self._splitRows(availHeight - sum(self._rowHeights[hi:n]), doInRowSplit=doInRowSplit)
        elif n < lo:
            return []
    repeatCols = self.repeatCols
    if not doInRowSplit:
        T = self
        data = self._cellvalues
    else:
        data = [_[:] for _ in self._cellvalues]
        if self._minRowHeights and availHeight < self._minRowHeights[n]:
            return []
        usedHeights = sum(self._rowHeights[:n])
        cellvalues = self._cellvalues[n]
        cellStyles = self._cellStyles[n]
        cellWidths = self._colWidths
        curRowHeight = self._rowHeights[n]
        minSplit = 0
        maxSplit = 0
        maxHeight = 0
        for column, (value, style, width) in enumerate(zip(cellvalues, cellStyles, cellWidths)):
            if self._spanCmds and self._spanRanges.get((column, n), None) is None:
                continue
            if isinstance(value, (tuple, list)):
                w, height = self._listCellGeom(value, width, style)
                height += style.topPadding + style.bottomPadding
                if height > maxHeight:
                    maxHeight = height
            elif isinstance(value, str):
                rows = value.split('\n')
                lineHeight = 1.2 * style.fontsize
                height = lineHeight * len(rows) + style.topPadding + style.bottomPadding
                minSplit = max(minSplit, lineHeight + style.topPadding)
                maxSplit = max(maxSplit, lineHeight + style.bottomPadding)
                if height > maxHeight:
                    maxHeight = height
        if minSplit + maxSplit > curRowHeight or minSplit > availHeight - usedHeights:
            if not self._spanCmds:
                return []
            splitCells = set()
            for column in range(self._ncols):
                cell = (column, n)
                if cell in self._rowSpanCells and self._spanRanges.get((column, n), None) is None:
                    for cell, span in self._spanRanges.items():
                        if span is None:
                            continue
                        start_col, start_row, end_col, end_row = span
                        if column >= start_col and column <= end_col and (n > start_row) and (n <= end_row):
                            splitCells.add(cell)
                            break
            if not splitCells:
                return []
            spanCmds = []
            for cmd, (sc, sr), (ec, er) in self._spanCmds:
                if sc < 0:
                    sc += self._ncols
                if ec < 0:
                    ec += self._ncols
                if sr < 0:
                    sr += self._nrows
                if er < 0:
                    er += self._nrows
                spanCmds.append((cmd, (sc, sr), (ec, er)))
            newCellStyles = [_[:] for _ in self._cellStyles]
            bkgrndcmds = self._bkgrndcmds
            for cell in splitCells:
                span_sc, span_sr, span_ec, span_er = self._spanRanges[cell]
                spanRect = self._spanRects[cell]
                oldHeight = spanRect[3]
                newHeight = sum(self._rowHeights[span_sr:n])
                oldStyle = newCellStyles[span_sr][span_sc]
                res = self._splitCell(self._cellvalues[span_sr][span_sc], oldStyle, oldHeight, newHeight, width)
                if not res:
                    return []
                data[span_sr][span_sc] = res[0]
                data[n][span_sc] = res[1]
                newSpanCmds = []
                for cmd, start, end in spanCmds:
                    if (span_sc, span_sr) == start and (span_ec, span_er) == end:
                        if n - 1 > span_sr or span_sc != span_ec:
                            newSpanCmds.append((cmd, (span_sc, span_sr), (span_ec, n - 1)))
                        if n < span_er or span_sc != span_ec:
                            newSpanCmds.append((cmd, (span_sc, n), (span_ec, span_er)))
                    else:
                        newSpanCmds.append((cmd, start, end))
                spanCmds = newSpanCmds
                newbkgrndcmds = []
                for cmd, start, end, color in bkgrndcmds:
                    if start == (span_sc, span_sr):
                        newbkgrndcmds.append((cmd, start, (end[0], n - 1), color))
                        newbkgrndcmds.append((cmd, (start[0], n), (end[0], n), color))
                    else:
                        newbkgrndcmds.append((cmd, start, end, color))
                bkgrndcmds = newbkgrndcmds
                newStyle = oldStyle.copy()
                if oldStyle.valign == 'MIDDLE':
                    if res[0] and res[1]:
                        oldStyle.valign = 'BOTTOM'
                        newStyle.valign = 'TOP'
                    else:
                        h = self._listCellGeom(v[0] or v[1], width, oldStyle)[1]
                        margin = (curRowHeight - h) / 2
                        if v[0]:
                            oldStyle.topPadding += margin
                        elif v[1]:
                            newStyle.bottomPadding += margin
                newCellStyles[n][span_sc] = newStyle
            T = self.__class__(data, colWidths=self._colWidths, rowHeights=self._rowHeights, repeatRows=self.repeatRows, repeatCols=self.repeatCols, splitByRow=self.splitByRow, splitInRow=self.splitInRow, normalizedData=1, cellStyles=newCellStyles, ident=self.ident, spaceBefore=getattr(self, 'spaceBefore', None), longTableOptimize=self._longTableOptimize, cornerRadii=getattr(self, '_cornerRadii', None), renderCB=getattr(self, '_renderCB', None))
            T._bkgrndcmds = bkgrndcmds
            T._spanCmds = spanCmds
            T._nosplitCmds = self._nosplitCmds
            T._srflcmds = self._srflcmds
            T._sircmds = self._sircmds
            T._colpositions = self._colpositions
            T._rowpositions = self._rowpositions
            T._calcNoSplitRanges()
            T._calcSpanRanges()
            T._calcSpanRects()
            newlinecmds = []
            for linecmd in self._linecmds:
                op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space = linecmd
                if er < 0:
                    er += T._nrows
                if ec < 0:
                    ec += T._ncols
                if op == 'BOX' or (op == 'GRID' and (sr <= n or er >= n)) or (op == 'INNERGRID' and (sr < n or er > n)):
                    if op in ('GRID', 'INNERGRID'):
                        newlinecmds.append(('INNERGRID', (sc, sr), (ec, n - 1), weight, color, cap, dash, join, count, space))
                        newlinecmds.append(('INNERGRID', (sc, n), (ec, er), weight, color, cap, dash, join, count, space))
                        newlinecmds.append(('LINEBELOW', (sc, n - 1), (ec, n - 1), weight, color, cap, dash, join, count, space))
                    if op in ('GRID', 'BOX'):
                        newlinecmds.append(('LINEABOVE', (sc, sr), (ec, sr), weight, color, cap, dash, join, count, space))
                        newlinecmds.append(('LINEBELOW', (sc, er), (ec, er), weight, color, cap, dash, join, count, space))
                        newlinecmds.append(('LINEBEFORE', (sc, sr), (sc, er), weight, color, cap, dash, join, count, space))
                        newlinecmds.append(('LINEAFTER', (ec, sr), (ec, er), weight, color, cap, dash, join, count, space))
                else:
                    newlinecmds.append(linecmd)
                    continue
            for cell in splitCells:
                moddedcmds = []
                for linecmd in newlinecmds:
                    op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space = linecmd
                    span_sc, span_sr, span_ec, span_er = self._spanRanges[cell]
                    if (op == 'LINEABOVE' and er > span_sr and (sr <= span_er) or (op == 'LINEBELOW' and er >= span_sr and (sr < span_er))) and (sc <= span_ec and ec >= span_sc):
                        if op == 'LINEABOVE':
                            startrow = span_sr
                            endrow = span_er + 1
                        else:
                            startrow = span_sr - 1
                            endrow = span_er
                        if sr <= startrow:
                            moddedcmds.append((op, (sc, sr), (ec, startrow), weight, color, cap, dash, join, count, space))
                        if span_sc > sc:
                            moddedcmds.append((op, (sc, max(startrow, sr)), (span_sc - 1, min(er, endrow)), weight, color, cap, dash, join, count, space))
                        if span_ec < ec:
                            moddedcmds.append((op, (span_ec + 1, max(startrow, sr)), (ec, min(er, endrow)), weight, color, cap, dash, join, count, space))
                        if er >= endrow:
                            moddedcmds.append((op, (sc, endrow), (ec, er), weight, color, cap, dash, join, count, space))
                    else:
                        moddedcmds.append(linecmd)
                newlinecmds = moddedcmds
            T._linecmds = newlinecmds
            return T._splitRows(availHeight, doInRowSplit=False)
        splitPoint = min(availHeight - usedHeights, maxHeight - maxSplit)
        if splitPoint + 1 < self.splitInRow:
            return []
        remaining = self._height - splitPoint
        if remaining < self.splitInRow:
            return []
        R0 = []
        R0Height = 0
        R1 = []
        R1Height = 0
        R1Styles = []
        for value, style, width in zip(cellvalues, cellStyles, cellWidths):
            v = self._splitCell(value, style, curRowHeight, splitPoint, width)
            if not v:
                return []
            newStyle = style.copy()
            if style.valign == 'MIDDLE':
                if v[0] and v[1]:
                    style.valign = 'BOTTOM'
                    newStyle.valign = 'TOP'
                else:
                    h = self._listCellGeom(v[0] or v[1], width, style)[1]
                    margin = (curRowHeight - h) / 2
                    if v[0]:
                        style.topPadding += margin
                    elif v[1]:
                        newStyle.bottomPadding += margin
            R0.append(v[0])
            R1.append(v[1])
            h0 = self._listCellGeom(v[0], width, style)[1] + style.topPadding + style.bottomPadding
            R0Height = max(R0Height, h0)
            h1 = self._listCellGeom(v[1], width, style)[1] + style.topPadding + style.bottomPadding
            R1Height = max(R1Height, h1)
            R1Styles.append(newStyle)
        usedHeight = min(splitPoint, R0Height)
        newRowHeight = max(R1Height, self._rowHeights[n] - usedHeight)
        newRowHeights = self._rowHeights[:]
        newRowHeights.insert(n + 1, newRowHeight)
        newRowHeights[n] = usedHeight
        newCellStyles = self._cellStyles[:]
        newCellStyles.insert(n + 1, R1Styles)
        data = data[:n] + [R0] + [R1] + data[n + 1:]
        T = self.__class__(data, colWidths=self._colWidths, rowHeights=newRowHeights, repeatRows=self.repeatRows, repeatCols=self.repeatCols, splitByRow=self.splitByRow, splitInRow=self.splitInRow, normalizedData=1, cellStyles=newCellStyles, ident=self.ident, spaceBefore=getattr(self, 'spaceBefore', None), longTableOptimize=self._longTableOptimize, cornerRadii=getattr(self, '_cornerRadii', None), renderCB=getattr(self, '_renderCB', None))
        T._linecmds = self._stretchCommands(n, self._linecmds, lim)
        T._bkgrndcmds = self._stretchCommands(n, self._bkgrndcmds, lim)
        T._spanCmds = self._stretchCommands(n, self._spanCmds, lim)
        T._nosplitCmds = self._stretchCommands(n, self._nosplitCmds, lim)
        T._srflcmds = self._stretchCommands(n, self._srflcmds, lim)
        T._sircmds = self._stretchCommands(n, self._sircmds, lim)
        n = n + 1
    ident = self.ident
    if ident:
        ident = IdentStr(ident)
    lto = T._longTableOptimize
    if lto:
        splitH = T._rowHeights
    else:
        splitH = T._argH
    cornerRadii = getattr(self, '_cornerRadii', None)
    renderCB = getattr(self, '_renderCB', None)
    R0 = self.__class__(data[:n], colWidths=T._colWidths, rowHeights=splitH[:n], repeatRows=repeatRows, repeatCols=repeatCols, splitByRow=self.splitByRow, splitInRow=self.splitInRow, normalizedData=1, cellStyles=T._cellStyles[:n], ident=ident, spaceBefore=getattr(self, 'spaceBefore', None), longTableOptimize=lto, cornerRadii=cornerRadii[:2] if cornerRadii else None, renderCB=renderCB)
    nrows = T._nrows
    ncols = T._ncols
    _linecmds = T._splitLineCmds(n, doInRowSplit=doInRowSplit)
    R0._cr_0(n, _linecmds, nrows, doInRowSplit)
    R0._cr_0(n, T._bkgrndcmds, nrows, doInRowSplit, _srflMode=True)
    R0._cr_0(n, T._spanCmds, nrows, doInRowSplit)
    R0._cr_0(n, T._nosplitCmds, nrows, doInRowSplit)
    for c in T._srflcmds:
        R0._addCommand(c)
        if c[1][1] != 'splitlast':
            continue
        (sc, sr), (ec, er) = c[1:3]
        R0._addCommand((c[0],) + ((sc, n - 1), (ec, n - 1)) + tuple(c[3:]))
    if ident:
        ident = IdentStr(ident)
    if repeatRows:
        if isinstance(repeatRows, int):
            iRows = data[:repeatRows]
            iRowH = splitH[:repeatRows]
            iCS = T._cellStyles[:repeatRows]
            repeatRows = list(range(repeatRows))
        else:
            repeatRows = list(sorted(repeatRows))
            iRows = [data[i] for i in repeatRows]
            iRowH = [splitH[i] for i in repeatRows]
            iCS = [T._cellStyles[i] for i in repeatRows]
        R1 = self.__class__(iRows + data[n:], colWidths=T._colWidths, rowHeights=iRowH + splitH[n:], repeatRows=len(repeatRows), repeatCols=repeatCols, splitByRow=self.splitByRow, splitInRow=self.splitInRow, normalizedData=1, cellStyles=iCS + T._cellStyles[n:], ident=ident, spaceAfter=getattr(self, 'spaceAfter', None), longTableOptimize=lto, cornerRadii=cornerRadii, renderCB=renderCB)
        R1._cr_1_1(n, nrows, repeatRows, _linecmds, doInRowSplit)
        R1._cr_1_1(n, nrows, repeatRows, T._bkgrndcmds, doInRowSplit, _srflMode=True)
        R1._cr_1_1(n, nrows, repeatRows, T._spanCmds, doInRowSplit)
        R1._cr_1_1(n, nrows, repeatRows, T._nosplitCmds, doInRowSplit)
    else:
        R1 = self.__class__(data[n:], colWidths=T._colWidths, rowHeights=splitH[n:], repeatRows=repeatRows, repeatCols=repeatCols, splitByRow=self.splitByRow, splitInRow=self.splitInRow, normalizedData=1, cellStyles=T._cellStyles[n:], ident=ident, spaceAfter=getattr(self, 'spaceAfter', None), longTableOptimize=lto, cornerRadii=[0, 0] + cornerRadii[2:] if cornerRadii else None, renderCB=renderCB)
        R1._cr_1_0(n, _linecmds, doInRowSplit)
        R1._cr_1_0(n, T._bkgrndcmds, doInRowSplit, _srflMode=True)
        R1._cr_1_0(n, T._spanCmds, doInRowSplit)
        R1._cr_1_0(n, T._nosplitCmds, doInRowSplit)
    for c in T._srflcmds:
        R1._addCommand(c)
        if c[1][1] != 'splitfirst':
            continue
        (sc, sr), (ec, er) = c[1:3]
        R1._addCommand((c[0],) + ((sc, 0), (ec, 0)) + tuple(c[3:]))
    R0.hAlign = R1.hAlign = T.hAlign
    R0.vAlign = R1.vAlign = T.vAlign
    self.onSplit(R0)
    self.onSplit(R1)
    return [R0, R1]
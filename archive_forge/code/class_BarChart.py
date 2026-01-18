import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
class BarChart(PlotArea):
    """Abstract base class, unusable by itself."""
    _attrMap = AttrMap(BASE=PlotArea, useAbsolute=AttrMapValue(EitherOr((isBoolean, EitherOr((isString, isNumber)))), desc='Flag to use absolute spacing values; use string of gsb for finer control\n(g=groupSpacing,s=barSpacing,b=barWidth).', advancedUsage=1), barWidth=AttrMapValue(isNumber, desc='The width of an individual bar.'), groupSpacing=AttrMapValue(isNumber, desc='Width between groups of bars.'), barSpacing=AttrMapValue(isNumber, desc='Width between individual bars.'), bars=AttrMapValue(None, desc='Handle of the individual bars.'), valueAxis=AttrMapValue(None, desc='Handle of the value axis.'), categoryAxis=AttrMapValue(None, desc='Handle of the category axis.'), data=AttrMapValue(None, desc='Data to be plotted, list of (lists of) numbers.'), barLabels=AttrMapValue(None, desc='Handle to the list of bar labels.'), barLabelFormat=AttrMapValue(None, desc='Formatting string or function used for bar labels. Can be a list or list of lists of such.'), barLabelCallOut=AttrMapValue(None, desc='Callout function(label)\nlabel._callOutInfo = (self,g,rowNo,colNo,x,y,width,height,x00,y00,x0,y0)', advancedUsage=1), barLabelArray=AttrMapValue(None, desc='explicit array of bar label values, must match size of data if present.'), reversePlotOrder=AttrMapValue(isBoolean, desc='If true, reverse common category plot order.', advancedUsage=1), naLabel=AttrMapValue(NoneOrInstanceOfNA_Label, desc='Label to use for N/A values.', advancedUsage=1), annotations=AttrMapValue(None, desc='list of callables, will be called with self, xscale, yscale.'), categoryLabelBarSize=AttrMapValue(isNumber, desc='width to leave for a category label to go between categories.'), categoryLabelBarOrder=AttrMapValue(OneOf('first', 'last', 'auto'), desc='where any label bar should appear first/last'), barRecord=AttrMapValue(None, desc='callable(bar,label=labelText,value=value,**kwds) to record bar information', advancedUsage=1), zIndexOverrides=AttrMapValue(isStringOrNone, desc="None (the default ie use old z ordering scheme) or a ',' separated list of key=value (int/float) for new zIndex ordering. If used defaults are\n    background=0,\n    categoryAxis=1,\n    valueAxis=2,\n    bars=3,\n    barLabels=4,\n    categoryAxisGrid=5,\n    valueAxisGrid=6,\n    annotations=7"), categoryNALabel=AttrMapValue(NoneOrInstanceOfNA_Label, desc='Label to use for a group of N/A values.', advancedUsage=1), seriesOrder=AttrMapValue(SequenceOf(SequenceOf(isInt, emptyOK=0, NoneOK=0, lo=1), emptyOK=0, NoneOK=1, lo=1), "dynamic 'mixed' category style case"))

    def makeSwatchSample(self, rowNo, x, y, width, height):
        baseStyle = self.bars
        styleIdx = rowNo % len(baseStyle)
        style = baseStyle[styleIdx]
        strokeColor = getattr(style, 'strokeColor', getattr(baseStyle, 'strokeColor', None))
        fillColor = getattr(style, 'fillColor', getattr(baseStyle, 'fillColor', None))
        strokeDashArray = getattr(style, 'strokeDashArray', getattr(baseStyle, 'strokeDashArray', None))
        strokeWidth = getattr(style, 'strokeWidth', getattr(style, 'strokeWidth', None))
        swatchMarker = getattr(style, 'swatchMarker', getattr(baseStyle, 'swatchMarker', None))
        if swatchMarker:
            return uSymbol2Symbol(swatchMarker, x + width / 2.0, y + height / 2.0, fillColor)
        elif getattr(style, 'isLine', False):
            yh2 = y + height / 2.0
            if hasattr(style, 'symbol'):
                S = style.symbol
            elif hasattr(baseStyle, 'symbol'):
                S = baseStyle.symbol
            else:
                S = None
            L = Line(x, yh2, x + width, yh2, strokeColor=style.strokeColor or style.fillColor, strokeWidth=style.strokeWidth, strokeDashArray=style.strokeDashArray)
            if S:
                S = uSymbol2Symbol(S, x + width / 2.0, yh2, style.strokeColor or style.fillColor)
            if S and L:
                g = Group()
                g.add(L)
                g.add(S)
                return g
            return S or L
        else:
            return Rect(x, y, width, height, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray, fillColor=fillColor)

    def getSeriesName(self, i, default=None):
        """return series name i or default"""
        return _objStr(getattr(self.bars[i], 'name', default))

    def __init__(self):
        assert self.__class__.__name__ not in ('BarChart', 'BarChart3D'), 'Abstract Class %s Instantiated' % self.__class__.__name__
        if self._flipXY:
            self.categoryAxis = YCategoryAxis()
            self.valueAxis = XValueAxis()
        else:
            self.categoryAxis = XCategoryAxis()
            self.valueAxis = YValueAxis()
        self.categoryAxis._attrMap['style'].validate = OneOf('stacked', 'parallel', 'parallel_3d', 'mixed')
        PlotArea.__init__(self)
        self.barSpacing = 0
        self.reversePlotOrder = 0
        self.data = [(100, 110, 120, 130), (70, 80, 85, 90)]
        self.useAbsolute = 0
        self.barWidth = 10
        self.groupSpacing = 5
        self.barSpacing = 0
        self.barLabels = TypedPropertyCollection(BarChartLabel)
        self.barLabels.boxAnchor = 'c'
        self.barLabels.textAnchor = 'middle'
        self.barLabelFormat = None
        self.barLabelArray = None
        self.barLabels.nudge = 0
        self.bars = TypedPropertyCollection(BarChartProperties)
        self.bars.strokeWidth = 1
        self.bars.strokeColor = colors.black
        self.bars.strokeDashArray = None
        self.bars[0].fillColor = colors.red
        self.bars[1].fillColor = colors.green
        self.bars[2].fillColor = colors.blue
        self.naLabel = self.categoryNALabel = None
        self.zIndexOverrides = None

    def demo(self):
        """Shows basic use of a bar chart"""
        if self.__class__.__name__ == 'BarChart':
            raise NotImplementedError('Abstract Class BarChart has no demo')
        drawing = Drawing(200, 100)
        bc = self.__class__()
        drawing.add(bc)
        return drawing

    def getSeriesOrder(self):
        bs = getattr(self, 'seriesOrder', None)
        n = len(self.data)
        if not bs:
            R = [(ss,) for ss in range(n)]
        else:
            bars = self.bars
            unseen = set(range(n))
            lines = set()
            R = []
            for s in bs:
                g = {ss for ss in s if 0 <= ss <= n}
                gl = {ss for ss in g if bars.checkAttr(ss, 'isLine', False)}
                if gl:
                    g -= gl
                    lines |= gl
                    unseen -= gl
                if g:
                    R.append(tuple(g))
                    unseen -= g
            if unseen:
                R.extend(((ss,) for ss in sorted(unseen)))
            if lines:
                R.extend(((ss,) for ss in sorted(lines)))
        self._seriesOrder = R

    def _getConfigureData(self):
        cAStyle = self.categoryAxis.style
        data = self.data
        cc = max(list(map(len, data)))
        _data = data
        if cAStyle not in ('parallel', 'parallel_3d'):
            data = []

            def _accumulate(*D):
                pdata = max((len(d) for d in D)) * [0]
                ndata = pdata[:]
                for d in D:
                    for i, v in enumerate(d):
                        v = v or 0
                        if v <= -1e-06:
                            ndata[i] += v
                        else:
                            pdata[i] += v
                data.append(ndata)
                data.append(pdata)
            if cAStyle == 'stacked':
                _accumulate(*_data)
            else:
                self.getSeriesOrder()
                for b in self._seriesOrder:
                    _accumulate(*(_data[j] for j in b))
        self._configureData = data

    def _getMinMax(self):
        """Attempt to return the data range"""
        self._getConfigureData()
        self.valueAxis._setRange(self._configureData)
        return (self.valueAxis._valueMin, self.valueAxis._valueMax)

    def _drawBegin(self, org, length):
        """Position and configure value axis, return crossing value"""
        vA = self.valueAxis
        vA.setPosition(self.x, self.y, length)
        self._getConfigureData()
        vA.configure(self._configureData)
        crossesAt = vA.scale(0)
        return crossesAt if vA.forceZero or (crossesAt >= org and crossesAt <= org + length) else org

    def _drawFinish(self):
        """finalize the drawing of a barchart"""
        cA = self.categoryAxis
        vA = self.valueAxis
        cA.configure(self._configureData)
        self.calcBarPositions()
        g = Group()
        zIndex = getattr(self, 'zIndexOverrides', None)
        if not zIndex:
            g.add(self.makeBackground())
            cAdgl = getattr(cA, 'drawGridLast', False)
            vAdgl = getattr(vA, 'drawGridLast', False)
            if not cAdgl:
                cA.makeGrid(g, parent=self, dim=vA.getGridDims)
            if not vAdgl:
                vA.makeGrid(g, parent=self, dim=cA.getGridDims)
            g.add(self.makeBars())
            g.add(cA)
            g.add(vA)
            if cAdgl:
                cA.makeGrid(g, parent=self, dim=vA.getGridDims)
            if vAdgl:
                vA.makeGrid(g, parent=self, dim=cA.getGridDims)
            for a in getattr(self, 'annotations', ()):
                g.add(a(self, cA.scale, vA.scale))
        else:
            Z = dict(background=0, categoryAxis=1, valueAxis=2, bars=3, barLabels=4, categoryAxisGrid=5, valueAxisGrid=6, annotations=7)
            for z in zIndex.strip().split(','):
                z = z.strip()
                if not z:
                    continue
                try:
                    k, v = z.split('=')
                except:
                    raise ValueError('Badly formatted zIndex clause %r in %r\nallowed variables are\n%s' % (z, zIndex, '\n'.join(['%s=%r' % (k, Z[k]) for k in sorted(Z.keys())])))
                if k not in Z:
                    raise ValueError('Unknown zIndex variable %r in %r\nallowed variables are\n%s' % (k, Z, '\n'.join(['%s=%r' % (k, Z[k]) for k in sorted(Z.keys())])))
                try:
                    v = literal_eval(v)
                    assert isinstance(v, (float, int))
                except:
                    raise ValueError('Bad zIndex value %r in clause %r of zIndex\nallowed variables are\n%s' % (v, z, zIndex, '\n'.join(['%s=%r' % (k, Z[k]) for k in sorted(Z.keys())])))
                Z[k] = v
            Z = [(v, k) for k, v in Z.items()]
            Z.sort()
            b = self.makeBars()
            bl = b.contents.pop(-1)
            for v, k in Z:
                if k == 'background':
                    g.add(self.makeBackground())
                elif k == 'categoryAxis':
                    g.add(cA)
                elif k == 'categoryAxisGrid':
                    cA.makeGrid(g, parent=self, dim=vA.getGridDims)
                elif k == 'valueAxis':
                    g.add(vA)
                elif k == 'valueAxisGrid':
                    vA.makeGrid(g, parent=self, dim=cA.getGridDims)
                elif k == 'bars':
                    g.add(b)
                elif k == 'barLabels':
                    g.add(bl)
                elif k == 'annotations':
                    for a in getattr(self, 'annotations', ()):
                        g.add(a(self, cA.scale, vA.scale))
        del self._configureData
        return g

    def calcBarPositions(self):
        """Works out where they go. default vertical.

        Sets an attribute _barPositions which is a list of
        lists of (x, y, width, height) matching the data.
        """
        flipXY = self._flipXY
        if flipXY:
            org = self.y
        else:
            org = self.x
        cA = self.categoryAxis
        cScale = cA.scale
        data = self.data
        seriesCount = self._seriesCount = len(data)
        self._rowLength = rowLength = max(list(map(len, data)))
        wG = self.groupSpacing
        barSpacing = self.barSpacing
        barWidth = self.barWidth
        clbs = getattr(self, 'categoryLabelBarSize', 0)
        clbo = getattr(self, 'categoryLabelBarOrder', 'auto')
        if clbo == 'auto':
            clbo = flipXY and 'last' or 'first'
        clbo = clbo == 'first'
        style = cA.style
        bars = self.bars
        lineCount = sum((int(bars.checkAttr(_, 'isLine', False)) for _ in range(seriesCount)))
        seriesMLineCount = seriesCount - lineCount
        if style == 'mixed':
            ss = self._seriesOrder
            barsPerGroup = len(ss) - lineCount
            wB = barsPerGroup * barWidth
            wS = (barsPerGroup - 1) * barSpacing
            if barsPerGroup > 1:
                bGapB = barWidth
                bGapS = barSpacing
            else:
                bGapB = bGapS = 0
            accumNeg = barsPerGroup * rowLength * [0]
            accumPos = accumNeg[:]
        elif style in ('parallel', 'parallel_3d'):
            barsPerGroup = 1
            wB = seriesMLineCount * barWidth
            wS = (seriesMLineCount - 1) * barSpacing
            bGapB = barWidth
            bGapS = barSpacing
        else:
            barsPerGroup = seriesMLineCount
            accumNeg = rowLength * [0]
            accumPos = accumNeg[:]
            wB = barWidth
            wS = bGapB = bGapS = 0
        self._groupWidth = groupWidth = wG + wB + wS
        useAbsolute = self.useAbsolute
        if useAbsolute:
            if not isinstance(useAbsolute, str):
                useAbsolute = 7
            else:
                useAbsolute = 0 + 1 * ('b' in useAbsolute) + 2 * ('g' in useAbsolute) + 4 * ('s' in useAbsolute)
        else:
            useAbsolute = 0
        aW0 = float(cScale(0)[1])
        aW = aW0 - clbs
        if useAbsolute == 0:
            self._normFactor = fB = fG = fS = aW / groupWidth
        elif useAbsolute == 7:
            fB = fG = fS = 1.0
            _cscale = cA._scale
        elif useAbsolute == 1:
            fB = 1.0
            fG = fS = (aW - wB) / (wG + wS)
        elif useAbsolute == 2:
            fG = 1.0
            fB = fS = (aW - wG) / (wB + wS)
        elif useAbsolute == 3:
            fB = fG = 1.0
            fS = (aW - wG - wB) / wS if wS else 0
        elif useAbsolute == 4:
            fS = 1.0
            fG = fB = (aW - wS) / (wG + wB)
        elif useAbsolute == 5:
            fS = fB = 1.0
            fG = (aW - wB - wS) / wG
        elif useAbsolute == 6:
            fS = fG = 1
            fB = (aW - wS - wG) / wB
        self._normFactorB = fB
        self._normFactorG = fG
        self._normFactorS = fS
        vA = self.valueAxis
        vScale = vA.scale
        vARD = vA.reverseDirection
        vm, vM = (vA._valueMin, vA._valueMax)
        if vm <= 0 <= vM:
            baseLine = vScale(0)
        elif 0 < vm:
            baseLine = vScale(vm)
        elif vM < 0:
            baseLine = vScale(vM)
        self._baseLine = baseLine
        width = barWidth * fB
        offs = 0.5 * wG * fG
        bGap = bGapB * fB + bGapS * fS
        if clbs:
            if clbo:
                lbpf = (offs + clbs / 6.0) / aW0
                offs += clbs
            else:
                lbpf = (offs + wB * fB + wS * fS + clbs / 6.0) / aW0
            cA.labels.labelPosFrac = lbpf
        self._barPositions = []
        aBP = self._barPositions.append
        reversePlotOrder = self.reversePlotOrder

        def _addBar(colNo, accx):
            if useAbsolute == 7:
                x = groupWidth * _cscale(colNo) + xVal + org
            else:
                g, _ = cScale(colNo)
                x = g + xVal
            datum = row[colNo]
            if datum is None:
                height = None
                y = baseLine
            else:
                if style not in ('parallel', 'parallel_3d') and (not isLine):
                    if datum <= -1e-06:
                        y = vScale(accumNeg[accx])
                        if y < baseLine if vARD else y > baseLine:
                            y = baseLine
                        accumNeg[accx] += datum
                        datum = accumNeg[accx]
                    else:
                        y = vScale(accumPos[accx])
                        if y > baseLine if vARD else y < baseLine:
                            y = baseLine
                        accumPos[accx] += datum
                        datum = accumPos[accx]
                else:
                    y = baseLine
                height = vScale(datum) - y
                if -1e-08 < height <= 1e-08:
                    height = 1e-08
                    if datum < -1e-08:
                        height = -1e-08
            barRow.append(flipXY and (y, x, height, width) or (x, y, width, height))
        if style != 'mixed':
            lineSeen = 0
            for rowNo, row in enumerate(data):
                barRow = []
                xVal = barsPerGroup - 1 - rowNo if reversePlotOrder else rowNo
                xVal = offs + xVal * bGap
                isLine = bars.checkAttr(rowNo, 'isLine', False)
                if isLine:
                    lineSeen += 1
                    xVal = offs + (seriesMLineCount - 1) * bGap * 0.5
                else:
                    xVal -= lineSeen * bGap
                for colNo in range(rowLength):
                    _addBar(colNo, colNo)
                aBP(barRow)
        else:
            lineSeen = 0
            for sb, sg in enumerate(self._seriesOrder):
                style = 'parallel' if len(sg) <= 1 else 'stacked'
                for rowNo in sg:
                    xVal = barsPerGroup - 1 - sb if reversePlotOrder else sb
                    xVal = offs + xVal * bGap
                    barRow = []
                    row = data[rowNo]
                    isLine = bars.checkAttr(rowNo, 'isLine', False)
                    if isLine:
                        lineSeen += 1
                        xVal = offs + (barsPerGroup - 1) * bGap * 0.5
                    else:
                        xVal -= lineSeen * bGap
                    for colNo in range(rowLength):
                        _addBar(colNo, colNo * barsPerGroup + sb)
                    aBP(barRow)

    def _getLabelText(self, rowNo, colNo):
        """return formatted label text"""
        labelFmt = self.barLabelFormat
        if isinstance(labelFmt, (list, tuple)):
            labelFmt = labelFmt[rowNo]
            if isinstance(labelFmt, (list, tuple)):
                labelFmt = labelFmt[colNo]
        if labelFmt is None:
            labelText = None
        elif labelFmt == 'values':
            labelText = self.barLabelArray[rowNo][colNo]
        elif isStr(labelFmt):
            labelText = labelFmt % self.data[rowNo][colNo]
        elif hasattr(labelFmt, '__call__'):
            labelText = labelFmt(self.data[rowNo][colNo])
        else:
            msg = 'Unknown formatter type %s, expected string or function' % labelFmt
            raise Exception(msg)
        return labelText

    def _labelXY(self, label, x, y, width, height):
        """Compute x, y for a label"""
        nudge = label.nudge
        bt = getattr(label, 'boxTarget', 'normal')
        anti = bt == 'anti'
        if anti:
            nudge = -nudge
        pm = value = height
        if anti:
            value = 0
        a = x + 0.5 * width
        nudge = (height >= 0 and 1 or -1) * nudge
        if bt == 'mid':
            b = y + height * 0.5
        elif bt == 'hi':
            if value >= 0:
                b = y + value + nudge
            else:
                b = y - nudge
                pm = -pm
        elif bt == 'lo':
            if value <= 0:
                b = y + value + nudge
            else:
                b = y - nudge
                pm = -pm
        else:
            b = y + value + nudge
        label._pmv = pm
        return (a, b, pm)

    def _addBarLabel(self, g, rowNo, colNo, x, y, width, height):
        text = self._getLabelText(rowNo, colNo)
        if text:
            self._addLabel(text, self.barLabels[rowNo, colNo], g, rowNo, colNo, x, y, width, height)

    def _addNABarLabel(self, g, rowNo, colNo, x, y, width, height, calcOnly=False, na=None):
        if na is None:
            na = self.naLabel
        if na and na.text:
            na = copy.copy(na)
            v = self.valueAxis._valueMax <= 0 and -1e-08 or 1e-08
            if width is None:
                width = v
            if height is None:
                height = v
            return self._addLabel(na.text, na, g, rowNo, colNo, x, y, width, height, calcOnly=calcOnly)

    def _addLabel(self, text, label, g, rowNo, colNo, x, y, width, height, calcOnly=False):
        if label.visible:
            labelWidth = stringWidth(text, label.fontName, label.fontSize)
            flipXY = self._flipXY
            if flipXY:
                y0, x0, pm = self._labelXY(label, y, x, height, width)
            else:
                x0, y0, pm = self._labelXY(label, x, y, width, height)
            fixedEnd = getattr(label, 'fixedEnd', None)
            if fixedEnd is not None:
                v = fixedEnd._getValue(self, pm)
                x00, y00 = (x0, y0)
                if flipXY:
                    x0 = v
                else:
                    y0 = v
            elif flipXY:
                x00 = x0
                y00 = y + height / 2.0
            else:
                x00 = x + width / 2.0
                y00 = y0
            fixedStart = getattr(label, 'fixedStart', None)
            if fixedStart is not None:
                v = fixedStart._getValue(self, pm)
                if flipXY:
                    x00 = v
                else:
                    y00 = v
            if pm < 0:
                if flipXY:
                    dx = -2 * label.dx
                    dy = 0
                else:
                    dy = -2 * label.dy
                    dx = 0
            else:
                dy = dx = 0
            if calcOnly:
                return (x0 + dx, y0 + dy)
            label.setOrigin(x0 + dx, y0 + dy)
            label.setText(text)
            sC, sW = (label.lineStrokeColor, label.lineStrokeWidth)
            if sC and sW:
                g.insert(0, Line(x00, y00, x0, y0, strokeColor=sC, strokeWidth=sW))
            g.add(label)
            alx = getattr(self, 'barLabelCallOut', None)
            if alx:
                label._callOutInfo = (self, g, rowNo, colNo, x, y, width, height, x00, y00, x0, y0)
                alx(label)
                del label._callOutInfo

    def _makeBar(self, g, x, y, width, height, rowNo, style):
        r = Rect(x, y, width, height)
        r.strokeWidth = style.strokeWidth
        r.fillColor = style.fillColor
        r.strokeColor = style.strokeColor
        if style.strokeDashArray:
            r.strokeDashArray = style.strokeDashArray
        g.add(r)

    def _makeBars(self, g, lg):
        bars = self.bars
        br = getattr(self, 'barRecord', None)
        BP = self._barPositions
        flipXY = self._flipXY
        catNAL = self.categoryNALabel
        catNNA = {}
        if catNAL:
            CBL = []
            rowNoL = len(self.data) - 1
            for rowNo, row in enumerate(BP):
                for colNo, (x, y, width, height) in enumerate(row):
                    if None not in (width, height):
                        catNNA[colNo] = 1
        lines = [].append
        lineSyms = [].append
        for rowNo, row in enumerate(BP):
            styleCount = len(bars)
            styleIdx = rowNo % styleCount
            rowStyle = bars[styleIdx]
            isLine = bars.checkAttr(rowNo, 'isLine', False)
            linePts = [].append
            for colNo, (x, y, width, height) in enumerate(row):
                style = (styleIdx, colNo) in bars and bars[styleIdx, colNo] or rowStyle
                if None in (width, height):
                    if not catNAL or colNo in catNNA:
                        self._addNABarLabel(lg, rowNo, colNo, x, y, width, height)
                    elif catNAL and colNo not in CBL:
                        r0 = self._addNABarLabel(lg, rowNo, colNo, x, y, width, height, True, catNAL)
                        if r0:
                            x, y, width, height = BP[rowNoL][colNo]
                            r1 = self._addNABarLabel(lg, rowNoL, colNo, x, y, width, height, True, catNAL)
                            x = (r0[0] + r1[0]) / 2.0
                            y = (r0[1] + r1[1]) / 2.0
                            self._addNABarLabel(lg, rowNoL, colNo, x, y, 0.0001, 0.0001, na=catNAL)
                        CBL.append(colNo)
                    if isLine:
                        linePts(None)
                    continue
                symbol = None
                if hasattr(style, 'symbol'):
                    symbol = copy.deepcopy(style.symbol)
                elif hasattr(self.bars, 'symbol'):
                    symbol = self.bars.symbol
                minDimen = getattr(style, 'minDimen', None)
                if minDimen:
                    if flipXY:
                        if width < 0:
                            width = min(-style.minDimen, width)
                        else:
                            width = max(style.minDimen, width)
                    elif height < 0:
                        height = min(-style.minDimen, height)
                    else:
                        height = max(style.minDimen, height)
                if isLine:
                    if not flipXY:
                        yL = y + height
                        xL = x + width * 0.5
                    else:
                        xL = x + width
                        yL = y + height * 0.5
                    linePts(xL)
                    linePts(yL)
                    if symbol:
                        sym = uSymbol2Symbol(tpcGetItem(symbol, colNo), xL, yL, style.strokeColor or style.fillColor)
                        if sym:
                            lineSyms(sym)
                elif symbol:
                    symbol.x = x
                    symbol.y = y
                    symbol.width = width
                    symbol.height = height
                    g.add(symbol)
                elif abs(width) > 1e-07 and abs(height) >= 1e-07 and (style.fillColor is not None or style.strokeColor is not None):
                    self._makeBar(g, x, y, width, height, rowNo, style)
                    if br:
                        br(g.contents[-1], label=self._getLabelText(rowNo, colNo), value=self.data[rowNo][colNo], rowNo=rowNo, colNo=colNo)
                self._addBarLabel(lg, rowNo, colNo, x, y, width, height)
            for linePts in yieldNoneSplits(linePts.__self__):
                if linePts:
                    lines(PolyLine(linePts, strokeColor=rowStyle.strokeColor or rowStyle.fillColor, strokeWidth=rowStyle.strokeWidth, strokeDashArray=rowStyle.strokeDashArray))
        for pl in lines.__self__:
            g.add(pl)
        for sym in lineSyms.__self__:
            g.add(sym)

    def _computeLabelPosition(self, text, label, rowNo, colNo, x, y, width, height):
        if label.visible:
            labelWidth = stringWidth(text, label.fontName, label.fontSize)
            flipXY = self._flipXY
            if flipXY:
                y0, x0, pm = self._labelXY(label, y, x, height, width)
            else:
                x0, y0, pm = self._labelXY(label, x, y, width, height)
            fixedEnd = getattr(label, 'fixedEnd', None)
            if fixedEnd is not None:
                v = fixedEnd._getValue(self, pm)
                x00, y00 = (x0, y0)
                if flipXY:
                    x0 = v
                else:
                    y0 = v
            elif flipXY:
                x00 = x0
                y00 = y + height / 2.0
            else:
                x00 = x + width / 2.0
                y00 = y0
            fixedStart = getattr(label, 'fixedStart', None)
            if fixedStart is not None:
                v = fixedStart._getValue(self, pm)
                if flipXY:
                    x00 = v
                else:
                    y00 = v
            if pm < 0:
                if flipXY:
                    dx = -2 * label.dx
                    dy = 0
                else:
                    dy = -2 * label.dy
                    dx = 0
            else:
                dy = dx = 0
            label.setOrigin(x0 + dx, y0 + dy)
            label.setText(text)
            return (pm, label.getBounds())

    def _computeBarPositions(self):
        """Information function, can be called by charts which want to with space around bars"""
        cA, vA = (self.categoryAxis, self.valueAxis)
        if vA:
            vA.joinAxis = cA
        if cA:
            cA.joinAxis = vA
        if self._flipXY:
            cA.setPosition(self._drawBegin(self.x, self.width), self.y, self.height)
        else:
            cA.setPosition(self.x, self._drawBegin(self.y, self.height), self.width)
        cA.configure(self._configureData)
        self.calcBarPositions()

    def _computeMaxSpace(self, size, required):
        """helper for madmen who want to put stuff inside their barcharts
        basically after _computebarPositions we slide a line of length size
        down the bar profile on either side of the bars to find the
        maximum space. If the space at any point is >= required then we're
        done. Otherwise we return the largest space location and amount.
        """
        flipXY = self._flipXY
        self._computeBarPositions()
        lenData = len(self.data)
        BP = self._barPositions
        C = []
        aC = C.append
        if flipXY:
            lo = self.x
            hi = lo + self.width
            end = self.y + self.height
            for bp in BP:
                for x, y, w, h in bp:
                    v = x + w
                    z = y + h
                    aC((min(y, z), max(y, z), min(x, v) - lo, hi - max(x, v)))
        else:
            lo = self.y
            hi = lo + self.height
            end = self.x + self.width
            for bp in BP:
                for x, y, w, h in bp:
                    v = y + h
                    z = x + w
                    aC((min(x, z), max(x, z), min(y, v) - lo, hi - max(y, v)))
        C.sort()
        R = [C[0]]
        for c in C:
            r = R[-1]
            if r[0] < c[1] and c[0] < r[1]:
                R[-1] = (min(r[0], c[0]), max(r[1], c[1]), min(r[2], c[2]), min(r[3], c[3]))
            else:
                R.append(c)
        C = R
        maxS = -2147483647
        maxP = None
        nC = len(C)
        for i, ci in enumerate(C):
            v0 = ci[0]
            v1 = v0 + size
            if v1 > end:
                break
            j = i
            alo = ahi = 2147483647
            while j < nC and C[j][1] <= v1:
                alo = min(C[j][2], alo)
                ahi = min(C[j][3], ahi)
                j += 1
            if alo > ahi:
                if alo > maxS:
                    maxS = alo
                    maxP = flipXY and (lo, v0, lo + alo, v0 + size, 0) or (v0, lo, v0 + size, lo + alo, 0)
                    if maxS >= required:
                        break
            elif ahi > maxS:
                maxS = ahi
                maxP = flipXY and (hi - ahi, v0, hi, v0 + size, 1) or (v0, hi - ahi, v0 + size, hi, 1)
                if maxS >= required:
                    break
        return (maxS, maxP)

    def _computeSimpleBarLabelPositions(self):
        """Information function, can be called by charts which want to mess with labels"""
        cA, vA = (self.categoryAxis, self.valueAxis)
        if vA:
            vA.joinAxis = cA
        if cA:
            cA.joinAxis = vA
        if self._flipXY:
            cA.setPosition(self._drawBegin(self.x, self.width), self.y, self.height)
        else:
            cA.setPosition(self.x, self._drawBegin(self.y, self.height), self.width)
        cA.configure(self._configureData)
        self.calcBarPositions()
        bars = self.bars
        R = [].append
        BP = self._barPositions
        for rowNo, row in enumerate(BP):
            C = [].append
            for colNo, (x, y, width, height) in enumerate(row):
                if None in (width, height):
                    na = self.naLabel
                    if na and na.text:
                        na = copy.copy(na)
                        v = self.valueAxis._valueMax <= 0 and -1e-08 or 1e-08
                        if width is None:
                            width = v
                        if height is None:
                            height = v
                        C(self._computeLabelPosition(na.text, na, rowNo, colNo, x, y, width, height))
                    else:
                        C(None)
                else:
                    text = self._getLabelText(rowNo, colNo)
                    if text:
                        C(self._computeLabelPosition(text, self.barLabels[rowNo, colNo], rowNo, colNo, x, y, width, height))
                    else:
                        C(None)
            R(C.__self__)
        return R.__self__

    def makeBars(self):
        g = Group()
        lg = Group()
        self._makeBars(g, lg)
        g.add(lg)
        return g

    def _desiredCategoryAxisLength(self):
        """for dynamically computing the desired category axis length"""
        style = self.categoryAxis.style
        data = self.data
        n = len(data)
        m = max(list(map(len, data)))
        if style == 'parallel':
            groupWidth = (n - 1) * self.barSpacing + n * self.barWidth
        else:
            groupWidth = self.barWidth
        return m * (self.groupSpacing + groupWidth)

    def draw(self):
        cA, vA = (self.categoryAxis, self.valueAxis)
        if vA:
            vA.joinAxis = cA
        if cA:
            cA.joinAxis = vA
        if self._flipXY:
            cA.setPosition(self._drawBegin(self.x, self.width), self.y, self.height)
        else:
            cA.setPosition(self.x, self._drawBegin(self.y, self.height), self.width)
        return self._drawFinish()
from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
class ValueAxis(_AxisG):
    """Abstract value axis, unusable in itself."""
    _attrMap = AttrMap(forceZero=AttrMapValue(EitherOr((isBoolean, OneOf('near'))), desc='Ensure zero in range if true.'), visible=AttrMapValue(isBoolean, desc='Display entire object, if true.'), visibleAxis=AttrMapValue(isBoolean, desc='Display axis line, if true.'), visibleLabels=AttrMapValue(isBoolean, desc='Display axis labels, if true.'), visibleTicks=AttrMapValue(isBoolean, desc='Display axis ticks, if true.'), visibleGrid=AttrMapValue(isBoolean, desc='Display axis grid, if true.'), strokeWidth=AttrMapValue(isNumber, desc='Width of axis line and ticks.'), strokeColor=AttrMapValue(isColorOrNone, desc='Color of axis line and ticks.'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Dash array used for axis line.'), strokeLineCap=AttrMapValue(OneOf(0, 1, 2), desc='Line cap 0=butt, 1=round & 2=square'), strokeLineJoin=AttrMapValue(OneOf(0, 1, 2), desc='Line join 0=miter, 1=round & 2=bevel'), strokeMiterLimit=AttrMapValue(isNumber, desc='miter limit control miter line joins'), gridStrokeWidth=AttrMapValue(isNumber, desc='Width of grid lines.'), gridStrokeColor=AttrMapValue(isColorOrNone, desc='Color of grid lines.'), gridStrokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Dash array used for grid lines.'), gridStrokeLineCap=AttrMapValue(OneOf(0, 1, 2), desc='Grid Line cap 0=butt, 1=round & 2=square'), gridStrokeLineJoin=AttrMapValue(OneOf(0, 1, 2), desc='Grid Line join 0=miter, 1=round & 2=bevel'), gridStrokeMiterLimit=AttrMapValue(isNumber, desc='Grid miter limit control miter line joins'), gridStart=AttrMapValue(isNumberOrNone, desc='Start of grid lines wrt axis origin'), gridEnd=AttrMapValue(isNumberOrNone, desc='End of grid lines wrt axis origin'), drawGridLast=AttrMapValue(isBoolean, desc='if true draw gridlines after everything else.'), minimumTickSpacing=AttrMapValue(isNumber, desc='Minimum value for distance between ticks.'), maximumTicks=AttrMapValue(isNumber, desc='Maximum number of ticks.'), labels=AttrMapValue(None, desc='Handle of the axis labels.'), labelAxisMode=AttrMapValue(OneOf('high', 'low', 'axis'), desc='Like joinAxisMode, but for the axis labels'), labelTextFormat=AttrMapValue(None, desc='Formatting string or function used for axis labels.'), labelTextPostFormat=AttrMapValue(None, desc='Extra Formatting string.'), labelTextScale=AttrMapValue(isNumberOrNone, desc='Scaling for label tick values.'), valueMin=AttrMapValue(isNumberOrNone, desc='Minimum value on axis.'), valueMax=AttrMapValue(isNumberOrNone, desc='Maximum value on axis.'), valueStep=AttrMapValue(isNumberOrNone, desc='Step size used between ticks.'), valueSteps=AttrMapValue(isListOfNumbersOrNone, desc='List of step sizes used between ticks.'), avoidBoundFrac=AttrMapValue(EitherOr((isNumberOrNone, SequenceOf(isNumber, emptyOK=0, lo=2, hi=2))), desc='Fraction of interval to allow above and below.'), avoidBoundSpace=AttrMapValue(EitherOr((isNumberOrNone, SequenceOf(isNumber, emptyOK=0, lo=2, hi=2))), desc='Space to allow above and below.'), abf_ignore_zero=AttrMapValue(EitherOr((NoneOr(isBoolean), SequenceOf(isBoolean, emptyOK=0, lo=2, hi=2))), desc='Set to True to make the avoidBoundFrac calculations treat zero as non-special'), rangeRound=AttrMapValue(OneOf('none', 'both', 'ceiling', 'floor'), 'How to round the axis limits'), zrangePref=AttrMapValue(isNumberOrNone, desc='Zero range axis limit preference.'), style=AttrMapValue(OneOf('normal', 'stacked', 'parallel_3d'), 'How values are plotted!'), skipEndL=AttrMapValue(OneOf('none', 'start', 'end', 'both'), desc='Skip high/low tick labels'), origShiftIPC=AttrMapValue(isNumberOrNone, desc='Lowest label shift interval ratio.'), origShiftMin=AttrMapValue(isNumberOrNone, desc='Minimum amount to shift.'), origShiftSpecialValue=AttrMapValue(isNumberOrNone, desc='special value for shift'), tickAxisMode=AttrMapValue(OneOf('high', 'low', 'axis'), desc='Like joinAxisMode, but for the ticks'), reverseDirection=AttrMapValue(isBoolean, desc='If true reverse category direction.'), annotations=AttrMapValue(None, desc='list of annotations'), loLLen=AttrMapValue(isNumber, desc='extra line length before start of the axis'), hiLLen=AttrMapValue(isNumber, desc='extra line length after end of the axis'), subTickNum=AttrMapValue(isNumber, desc='Number of axis sub ticks, if >0'), subTickLo=AttrMapValue(isNumber, desc='sub tick down or left'), subTickHi=AttrMapValue(isNumber, desc='sub tick up or right'), visibleSubTicks=AttrMapValue(isBoolean, desc='Display axis sub ticks, if true.'), visibleSubGrid=AttrMapValue(isBoolean, desc='Display axis sub grid, if true.'), subGridStrokeWidth=AttrMapValue(isNumber, desc='Width of grid lines.'), subGridStrokeColor=AttrMapValue(isColorOrNone, desc='Color of grid lines.'), subGridStrokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Dash array used for grid lines.'), subGridStrokeLineCap=AttrMapValue(OneOf(0, 1, 2), desc='Grid Line cap 0=butt, 1=round & 2=square'), subGridStrokeLineJoin=AttrMapValue(OneOf(0, 1, 2), desc='Grid Line join 0=miter, 1=round & 2=bevel'), subGridStrokeMiterLimit=AttrMapValue(isNumber, desc='Grid miter limit control miter line joins'), subGridStart=AttrMapValue(isNumberOrNone, desc='Start of grid lines wrt axis origin'), subGridEnd=AttrMapValue(isNumberOrNone, desc='End of grid lines wrt axis origin'), tickStrokeWidth=AttrMapValue(isNumber, desc='Width of ticks if specified.'), subTickStrokeWidth=AttrMapValue(isNumber, desc='Width of sub ticks if specified.'), subTickStrokeColor=AttrMapValue(isColorOrNone, desc='Color of sub ticks if specified.'), tickStrokeColor=AttrMapValue(isColorOrNone, desc='Color of ticks if specified.'), keepTickLabelsInside=AttrMapValue(isBoolean, desc='Ensure tick labels do not project beyond bounds of axis if true'), skipGrid=AttrMapValue(OneOf('none', 'top', 'both', 'bottom'), 'grid lines to skip top bottom both none'), requiredRange=AttrMapValue(isNumberOrNone, desc='Minimum required value range.'), innerTickDraw=AttrMapValue(isNoneOrCallable, desc='Callable to replace _drawInnerTicks'), extraMinMaxValues=AttrMapValue(isListOfNumbersOrNone, desc='extra values to use in min max calculation'))

    def __init__(self, **kw):
        assert self.__class__.__name__ != 'ValueAxis', 'Abstract Class ValueAxis Instantiated'
        self._setKeywords(**kw)
        self._setKeywords(_configured=0, _x=50, _y=50, _length=100, visible=1, visibleAxis=1, visibleLabels=1, visibleTicks=1, visibleGrid=0, forceZero=0, strokeWidth=1, strokeColor=STATE_DEFAULTS['strokeColor'], strokeDashArray=STATE_DEFAULTS['strokeDashArray'], strokeLineJoin=STATE_DEFAULTS['strokeLineJoin'], strokeLineCap=STATE_DEFAULTS['strokeLineCap'], strokeMiterLimit=STATE_DEFAULTS['strokeMiterLimit'], gridStrokeWidth=0.25, gridStrokeColor=STATE_DEFAULTS['strokeColor'], gridStrokeDashArray=STATE_DEFAULTS['strokeDashArray'], gridStrokeLineJoin=STATE_DEFAULTS['strokeLineJoin'], gridStrokeLineCap=STATE_DEFAULTS['strokeLineCap'], gridStrokeMiterLimit=STATE_DEFAULTS['strokeMiterLimit'], gridStart=None, gridEnd=None, drawGridLast=False, visibleSubGrid=0, visibleSubTicks=0, subTickNum=0, subTickLo=0, subTickHi=0, subGridStrokeLineJoin=STATE_DEFAULTS['strokeLineJoin'], subGridStrokeLineCap=STATE_DEFAULTS['strokeLineCap'], subGridStrokeMiterLimit=STATE_DEFAULTS['strokeMiterLimit'], subGridStrokeWidth=0.25, subGridStrokeColor=STATE_DEFAULTS['strokeColor'], subGridStrokeDashArray=STATE_DEFAULTS['strokeDashArray'], subGridStart=None, subGridEnd=None, labels=TypedPropertyCollection(Label), keepTickLabelsInside=0, minimumTickSpacing=10, maximumTicks=7, _labelTextFormat=None, labelAxisMode='axis', labelTextFormat=None, labelTextPostFormat=None, labelTextScale=None, valueMin=None, valueMax=None, valueStep=None, avoidBoundFrac=None, avoidBoundSpace=None, abf_ignore_zero=False, rangeRound='none', zrangePref=0, style='normal', skipEndL='none', origShiftIPC=None, origShiftMin=None, origShiftSpecialValue=None, tickAxisMode='axis', reverseDirection=0, loLLen=0, hiLLen=0, requiredRange=0, extraMinMaxValues=None)
        self.labels.angle = 0

    def setPosition(self, x, y, length):
        self._x = float(x)
        self._y = float(y)
        self._length = float(length)

    def configure(self, dataSeries):
        """Let the axis configure its scale and range based on the data.

        Called after setPosition. Let it look at a list of lists of
        numbers determine the tick mark intervals.  If valueMin,
        valueMax and valueStep are configured then it
        will use them; if any of them are set to None it
        will look at the data and make some sensible decision.
        You may override this to build custom axes with
        irregular intervals.  It creates an internal
        variable self._values, which is a list of numbers
        to use in plotting.
        """
        self._setRange(dataSeries)
        self._configure_end()

    def _configure_end(self):
        self._calcTickmarkPositions()
        self._calcScaleFactor()
        self._configured = 1

    def _getValueStepAndTicks(self, valueMin, valueMax, cache={}):
        try:
            K = (valueMin, valueMax)
            r = cache[K]
        except:
            self._valueMin = valueMin
            self._valueMax = valueMax
            valueStep, T = self._calcStepAndTickPositions()
            r = cache[K] = (valueStep, T, valueStep * 1e-08)
        return r

    def _preRangeAdjust(self, valueMin, valueMax):
        rr = self.requiredRange
        if rr > 0:
            r = valueMax - valueMin
            if r < rr:
                m = 0.5 * (valueMax + valueMin)
                rr *= 0.5
                y1 = min(m - rr, valueMin)
                y2 = max(m + rr, valueMax)
                if valueMin >= 100 and y1 < 100:
                    y2 = y2 + 100 - y1
                    y1 = 100
                elif valueMin >= 0 and y1 < 0:
                    y2 = y2 - y1
                    y1 = 0
                valueMin = self._cValueMin = y1
                valueMax = self._cValueMax = y2
        return (valueMin, valueMax)

    def _setRange(self, dataSeries):
        """Set minimum and maximum axis values.

        The dataSeries argument is assumed to be a list of data
        vectors. Each vector is itself a list or tuple of numbers.

        Returns a min, max tuple.
        """
        oMin = valueMin = self.valueMin
        oMax = valueMax = self.valueMax
        if valueMin is None:
            valueMin = self._cValueMin = _findMin(dataSeries, self._dataIndex, 0, self.extraMinMaxValues)
        if valueMax is None:
            valueMax = self._cValueMax = _findMax(dataSeries, self._dataIndex, 0, self.extraMinMaxValues)
        if valueMin == valueMax:
            if valueMax == 0:
                if oMin is None and oMax is None:
                    zrp = getattr(self, 'zrangePref', 0)
                    if zrp > 0:
                        valueMax = zrp
                        valueMin = 0
                    elif zrp < 0:
                        valueMax = 0
                        valueMin = zrp
                    else:
                        valueMax = 0.01
                        valueMin = -0.01
                elif self.valueMin is None:
                    valueMin = -0.01
                else:
                    valueMax = 0.01
            elif valueMax > 0:
                valueMax = 1.2 * valueMax
                valueMin = 0.0
            else:
                valueMax = 0.0
                valueMin = 1.2 * valueMin
        if getattr(self, '_bubblePlot', None):
            bubbleMax = float(_findMax(dataSeries, 2, 0))
            frac = 0.25
            bubbleV = frac * (valueMax - valueMin)
            self._bubbleV = bubbleV
            self._bubbleMax = bubbleMax
            self._bubbleRadius = frac * self._length

            def special(T, x, func, bubbleV=bubbleV, bubbleMax=bubbleMax):
                try:
                    v = T[2]
                except IndexError:
                    v = bubbleMAx * 0.1
                bubbleV *= (v / bubbleMax) ** 0.5
                return func(T[x] + bubbleV, T[x] - bubbleV)
            if oMin is None:
                valueMin = self._cValueMin = _findMin(dataSeries, self._dataIndex, 0, special=special, extraMinMaxValues=self.extraMinMaxValues)
            if oMax is None:
                valueMax = self._cValueMax = _findMax(dataSeries, self._dataIndex, 0, special=special, extraMinMaxValues=self.extraMinMaxValues)
        valueMin, valueMax = self._preRangeAdjust(valueMin, valueMax)
        rangeRound = self.rangeRound
        cMin = valueMin
        cMax = valueMax
        forceZero = self.forceZero
        if forceZero:
            if forceZero == 'near':
                forceZero = min(abs(valueMin), abs(valueMax)) <= 5 * (valueMax - valueMin)
            if forceZero:
                if valueMax < 0:
                    valueMax = 0
                elif valueMin > 0:
                    valueMin = 0
        abf = self.avoidBoundFrac
        do_rr = not getattr(self, 'valueSteps', None)
        do_abf = abf and do_rr
        if not isSeq(abf):
            abf = (abf, abf)
        abfiz = getattr(self, 'abf_ignore_zero', False)
        if not isSeq(abfiz):
            abfiz = (abfiz, abfiz)
        do_rr = rangeRound != 'none' and do_rr
        if do_rr:
            rrn = rangeRound in ['both', 'floor']
            rrx = rangeRound in ['both', 'ceiling']
        else:
            rrn = rrx = 0
        abS = self.avoidBoundSpace
        do_abs = abS
        if do_abs:
            if not isSeq(abS):
                abS = (abS, abS)
        aL = float(self._length)
        go = do_rr or do_abf or do_abs
        cache = {}
        iter = 0
        while go and iter <= 10:
            iter += 1
            go = 0
            if do_abf or do_abs:
                valueStep, T, fuzz = self._getValueStepAndTicks(valueMin, valueMax, cache)
                if do_abf:
                    i0 = valueStep * abf[0]
                    i1 = valueStep * abf[1]
                else:
                    i0 = i1 = 0
                if do_abs:
                    sf = (valueMax - valueMin) / aL
                    i0 = max(i0, abS[0] * sf)
                    i1 = max(i1, abS[1] * sf)
                if rrn:
                    v = T[0]
                else:
                    v = valueMin
                u = cMin - i0
                if (abfiz[0] or abs(v) > fuzz) and v >= u + fuzz:
                    valueMin = u
                    go = 1
                if rrx:
                    v = T[-1]
                else:
                    v = valueMax
                u = cMax + i1
                if (abfiz[1] or abs(v) > fuzz) and v <= u - fuzz:
                    valueMax = u
                    go = 1
            if do_rr:
                valueStep, T, fuzz = self._getValueStepAndTicks(valueMin, valueMax, cache)
                if rrn:
                    if valueMin < T[0] - fuzz:
                        valueMin = T[0] - valueStep
                        go = 1
                    else:
                        go = valueMin >= T[0] + fuzz
                        valueMin = T[0]
                if rrx:
                    if valueMax > T[-1] + fuzz:
                        valueMax = T[-1] + valueStep
                        go = 1
                    else:
                        go = valueMax <= T[-1] - fuzz
                        valueMax = T[-1]
        if iter and (not go):
            self._computedValueStep = valueStep
        else:
            self._computedValueStep = None
        self._valueMin = valueMin
        self._valueMax = valueMax
        origShiftIPC = self.origShiftIPC
        origShiftMin = self.origShiftMin
        if origShiftMin is not None or origShiftIPC is not None:
            origShiftSpecialValue = self.origShiftSpecialValue
            self._calcValueStep()
            valueMax, valueMin = (self._valueMax, self._valueMin)
            if origShiftSpecialValue is None or abs(origShiftSpecialValue - valueMin) < 1e-06:
                if origShiftIPC:
                    m = origShiftIPC * self._valueStep
                else:
                    m = 0
                if origShiftMin:
                    m = max(m, (valueMax - valueMin) * origShiftMin / self._length)
                self._valueMin -= m
        self._rangeAdjust()

    def _pseudo_configure(self):
        self._valueMin = self.valueMin
        self._valueMax = self.valueMax
        if hasattr(self, '_subTickValues'):
            del self._subTickValues
        self._configure_end()

    def _rangeAdjust(self):
        """Override this if you want to alter the calculated range.

        E.g. if want a minumamum range of 30% or don't want 100%
        as the first point.
        """
        pass

    def _adjustAxisTicks(self):
        """Override if you want to put slack at the ends of the axis
        eg if you don't want the last tick to be at the bottom etc
        """
        pass

    def _calcScaleFactor(self):
        """Calculate the axis' scale factor.
        This should be called only *after* the axis' range is set.
        Returns a number.
        """
        self._scaleFactor = self._length / float(self._valueMax - self._valueMin)
        return self._scaleFactor

    def _calcStepAndTickPositions(self):
        valueStep = getattr(self, '_computedValueStep', None)
        if valueStep:
            del self._computedValueStep
            self._valueStep = valueStep
        else:
            self._calcValueStep()
            valueStep = self._valueStep
        valueMin = self._valueMin
        valueMax = self._valueMax
        fuzz = 1e-08 * valueStep
        rangeRound = self.rangeRound
        i0 = int(float(valueMin) / valueStep)
        v = i0 * valueStep
        if rangeRound in ('both', 'floor'):
            if v > valueMin + fuzz:
                i0 -= 1
        elif v < valueMin - fuzz:
            i0 += 1
        i1 = int(float(valueMax) / valueStep)
        v = i1 * valueStep
        if rangeRound in ('both', 'ceiling'):
            if v < valueMax - fuzz:
                i1 += 1
        elif v > valueMax + fuzz:
            i1 -= 1
        return (valueStep, [i * valueStep for i in range(i0, i1 + 1)])

    def _calcTickPositions(self):
        return self._calcStepAndTickPositions()[1]

    def _calcSubTicks(self):
        if not hasattr(self, '_tickValues'):
            self._pseudo_configure()
        otv = self._tickValues
        if not hasattr(self, '_subTickValues'):
            acn = self.__class__.__name__
            if acn[:11] == 'NormalDateX':
                iFuzz = 0
                dCnv = int
            else:
                iFuzz = 1e-08
                dCnv = lambda x: x
            OTV = [tv for tv in otv if getattr(tv, '_doSubTicks', 1)]
            T = [].append
            nst = int(self.subTickNum)
            i = len(OTV)
            if i < 2:
                self._subTickValues = []
            else:
                if i == 2:
                    dst = OTV[1] - OTV[0]
                elif i == 3:
                    dst = max(OTV[1] - OTV[0], OTV[2] - OTV[1])
                else:
                    i >>= 1
                    dst = OTV[i + 1] - OTV[i]
                fuzz = dst * iFuzz
                vn = self._valueMin + fuzz
                vx = self._valueMax - fuzz
                if OTV[0] > vn:
                    OTV.insert(0, OTV[0] - dst)
                if OTV[-1] < vx:
                    OTV.append(OTV[-1] + dst)
                dst /= float(nst + 1)
                for i, x in enumerate(OTV[:-1]):
                    for j in range(nst):
                        t = x + dCnv((j + 1) * dst)
                        if t <= vn or t >= vx:
                            continue
                        T(t)
                self._subTickValues = T.__self__
        self._tickValues = self._subTickValues
        return otv

    def _calcTickmarkPositions(self):
        """Calculate a list of tick positions on the axis.  Returns a list of numbers."""
        self._tickValues = getattr(self, 'valueSteps', None)
        if self._tickValues:
            return self._tickValues
        self._tickValues = self._calcTickPositions()
        self._adjustAxisTicks()
        return self._tickValues

    def _calcValueStep(self):
        """Calculate _valueStep for the axis or get from valueStep."""
        if self.valueStep is None:
            rawRange = self._valueMax - self._valueMin
            rawInterval = rawRange / min(float(self.maximumTicks - 1), float(self._length) / self.minimumTickSpacing)
            self._valueStep = nextRoundNumber(rawInterval)
        else:
            self._valueStep = self.valueStep

    def _allIntTicks(self):
        return _allInt(self._tickValues)

    def makeTickLabels(self):
        g = Group()
        if not self.visibleLabels:
            return g
        f = self._labelTextFormat
        if f is None:
            f = self.labelTextFormat or (self._allIntTicks() and '%.0f' or _defaultLabelFormatter)
        elif f is str and self._allIntTicks():
            f = '%.0f'
        elif hasattr(f, 'calcPlaces'):
            f.calcPlaces(self._tickValues)
        post = self.labelTextPostFormat
        scl = self.labelTextScale
        pos = [self._x, self._y]
        d = self._dataIndex
        pos[1 - d] = self._labelAxisPos()
        labels = self.labels
        if self.skipEndL != 'none':
            if self.isXAxis:
                sk = self._x
            else:
                sk = self._y
            if self.skipEndL == 'start':
                sk = [sk]
            else:
                sk = [sk, sk + self._length]
                if self.skipEndL == 'end':
                    del sk[0]
        else:
            sk = []
        nticks = len(self._tickValues)
        nticks1 = nticks - 1
        for i, tick in enumerate(self._tickValues):
            label = i - nticks
            if label in labels:
                label = labels[label]
            else:
                label = labels[i]
            if f and label.visible:
                v = self.scale(tick)
                if sk:
                    for skv in sk:
                        if abs(skv - v) < 1e-06:
                            v = None
                            break
                if v is not None:
                    if scl is not None:
                        t = tick * scl
                    else:
                        t = tick
                    if isinstance(f, str):
                        txt = f % t
                    elif isSeq(f):
                        if i < len(f):
                            txt = f[i]
                        else:
                            txt = ''
                    elif hasattr(f, '__call__'):
                        if isinstance(f, TickLabeller):
                            txt = f(self, t)
                        else:
                            txt = f(t)
                    else:
                        raise ValueError('Invalid labelTextFormat %s' % f)
                    if post:
                        txt = post % txt
                    pos[d] = v
                    label.setOrigin(*pos)
                    label.setText(txt)
                    if self.keepTickLabelsInside:
                        if isinstance(self, XValueAxis):
                            a_x = self._x
                            if not i:
                                x0, y0, x1, y1 = label.getBounds()
                                if x0 < a_x:
                                    label = label.clone(dx=label.dx + a_x - x0)
                            if i == nticks1:
                                a_x1 = a_x + self._length
                                x0, y0, x1, y1 = label.getBounds()
                                if x1 > a_x1:
                                    label = label.clone(dx=label.dx - x1 + a_x1)
                    g.add(label)
        return g

    def scale(self, value):
        """Converts a numeric value to a plotarea position.
        The chart first configures the axis, then asks it to
        """
        assert self._configured, 'Axis cannot scale numbers before it is configured'
        if value is None:
            value = 0
        org = (self._x, self._y)[self._dataIndex]
        sf = self._scaleFactor
        if self.reverseDirection:
            sf = -sf
            org += self._length
        return org + sf * (value - self._valueMin)
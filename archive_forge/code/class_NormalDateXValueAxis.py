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
class NormalDateXValueAxis(XValueAxis):
    """An X axis applying additional rules.

    Depending on the data and some built-in rules, the axis
    displays normalDate values as nicely formatted dates.

    The client chart should have NormalDate X values.
    """
    _attrMap = AttrMap(BASE=XValueAxis, bottomAxisLabelSlack=AttrMapValue(isNumber, desc='Fractional amount used to adjust label spacing'), niceMonth=AttrMapValue(isBoolean, desc="Flag for displaying months 'nicely'."), forceEndDate=AttrMapValue(isBoolean, desc='Flag for enforced displaying of last date value.'), forceFirstDate=AttrMapValue(isBoolean, desc='Flag for enforced displaying of first date value.'), forceDatesEachYear=AttrMapValue(isListOfDaysAndMonths, desc='List of dates in format "31-Dec",' + '"1-Jan".  If present they will always be used for tick marks in the current year, rather ' + 'than the dates chosen by the automatic algorithm. Hyphen compulsory, case of month optional.'), xLabelFormat=AttrMapValue(None, desc="Label format string (e.g. '{mm}/{yy}') or function."), dayOfWeekName=AttrMapValue(SequenceOf(isString, emptyOK=0, lo=7, hi=7), desc='Weekday names.'), monthName=AttrMapValue(SequenceOf(isString, emptyOK=0, lo=12, hi=12), desc='Month names.'), dailyFreq=AttrMapValue(isBoolean, desc='True if we are to assume daily data to be ticked at end of month.'), specifiedTickDates=AttrMapValue(NoneOr(SequenceOf(isNormalDate)), desc='Actual tick values to use; no calculations done'), specialTickClear=AttrMapValue(isBoolean, desc='clear rather than delete close ticks when forced first/end dates'), skipGrid=AttrMapValue(OneOf('none', 'top', 'both', 'bottom'), 'grid lines to skip top bottom both none'))
    _valueClass = normalDate.ND

    def __init__(self, **kw):
        XValueAxis.__init__(self, **kw)
        self.bottomAxisLabelSlack = 0.1
        self.niceMonth = 1
        self.forceEndDate = 0
        self.forceFirstDate = 0
        self.forceDatesEachYear = []
        self.dailyFreq = 0
        self.xLabelFormat = '{mm}/{yy}'
        self.dayOfWeekName = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.monthName = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        self.specialTickClear = 0
        self.valueSteps = self.specifiedTickDates = None

    def _scalar2ND(self, x):
        """Convert a scalar to a NormalDate value."""
        d = self._valueClass()
        d.normalize(x)
        return d

    def _dateFormatter(self, v):
        """Create a formatted label for some value."""
        if not isinstance(v, normalDate.NormalDate):
            v = self._scalar2ND(v)
        d, m = (normalDate._dayOfWeekName, normalDate._monthName)
        try:
            normalDate._dayOfWeekName, normalDate._monthName = (self.dayOfWeekName, self.monthName)
            return v.formatMS(self.xLabelFormat)
        finally:
            normalDate._dayOfWeekName, normalDate._monthName = (d, m)

    def _xAxisTicker(self, xVals):
        """Complex stuff...

        Needs explanation...

        Yes please says Andy :-(.  Modified on 19 June 2006 to attempt to allow
        a mode where one can specify recurring days and months.
        """
        VC = self._valueClass
        axisLength = self._length
        formatter = self._dateFormatter
        if isinstance(formatter, TickLabeller):

            def formatter(tick):
                return self._dateFormatter(self, tick)
        firstDate = xVals[0] if not self.valueMin else VC(self.valueMin)
        endDate = xVals[-1] if not self.valueMax else VC(self.valueMax)
        labels = self.labels
        fontName, fontSize, leading = (labels.fontName, labels.fontSize, labels.leading)
        textAnchor, boxAnchor, angle = (labels.textAnchor, labels.boxAnchor, labels.angle)
        RBL = _textBoxLimits(formatter(firstDate).split('\n'), fontName, fontSize, leading or 1.2 * fontSize, textAnchor, boxAnchor)
        RBL = _rotatedBoxLimits(RBL[0], RBL[1], RBL[2], RBL[3], angle)
        xLabelW = RBL[1] - RBL[0]
        xLabelH = RBL[3] - RBL[2]
        w = max(xLabelW, labels.width or 0, self.minimumTickSpacing)
        W = w + w * self.bottomAxisLabelSlack
        ticks = []
        labels = []
        maximumTicks = self.maximumTicks
        if self.specifiedTickDates:
            ticks = [VC(x) for x in self.specifiedTickDates]
            labels = [formatter(d) for d in ticks]
            if self.forceFirstDate and firstDate == ticks[0] and (axisLength / float(ticks[-1] - ticks[0]) * (ticks[1] - ticks[0]) <= W):
                if self.specialTickClear:
                    labels[1] = ''
                else:
                    del ticks[1], labels[1]
            if self.forceEndDate and endDate == ticks[-1] and (axisLength / float(ticks[-1] - ticks[0]) * (ticks[-1] - ticks[-2]) <= W):
                if self.specialTickClear:
                    labels[-2] = ''
                else:
                    del ticks[-2], labels[-2]
            return (ticks, labels)
        if self.forceDatesEachYear:
            forcedPartialDates = list(map(parseDayAndMonth, self.forceDatesEachYear))
            firstYear = firstDate.year()
            lastYear = endDate.year()
            ticks = []
            labels = []
            yyyy = firstYear
            while yyyy <= lastYear:
                for dd, mm in forcedPartialDates:
                    theDate = normalDate.ND((yyyy, mm, dd))
                    if theDate >= firstDate and theDate <= endDate:
                        ticks.append(theDate)
                        labels.append(formatter(theDate))
                yyyy += 1
            if self.forceFirstDate and firstDate != ticks[0]:
                ticks.insert(0, firstDate)
                labels.insert(0, formatter(firstDate))
                if axisLength / float(ticks[-1] - ticks[0]) * (ticks[1] - ticks[0]) <= W:
                    if self.specialTickClear:
                        labels[1] = ''
                    else:
                        del ticks[1], labels[1]
            if self.forceEndDate and endDate != ticks[-1]:
                ticks.append(endDate)
                labels.append(formatter(endDate))
                if axisLength / float(ticks[-1] - ticks[0]) * (ticks[-1] - ticks[-2]) <= W:
                    if self.specialTickClear:
                        labels[-2] = ''
                    else:
                        del ticks[-2], labels[-2]
            return (ticks, labels)

        def addTick(i, xVals=xVals, formatter=formatter, ticks=ticks, labels=labels):
            ticks.insert(0, xVals[i])
            labels.insert(0, formatter(xVals[i]))
        n = len(xVals)
        for d in _NDINTM:
            k = n / d
            if k <= maximumTicks and k * W <= axisLength:
                i = n - 1
                if self.niceMonth:
                    j = endDate.month() % (d <= 12 and d or 12)
                    if j:
                        if self.forceEndDate:
                            addTick(i)
                            ticks[0]._doSubTicks = 0
                        i -= j
                try:
                    wfd = firstDate.month() == xVals[1].month()
                except:
                    wfd = 0
                while i >= wfd:
                    addTick(i)
                    i -= d
                if self.forceFirstDate and ticks[0] != firstDate:
                    addTick(0)
                    ticks[0]._doSubTicks = 0
                    if axisLength / float(ticks[-1] - ticks[0]) * (ticks[1] - ticks[0]) <= W:
                        if self.specialTickClear:
                            labels[1] = ''
                        else:
                            del ticks[1], labels[1]
                if self.forceEndDate and self.niceMonth and j:
                    if axisLength / float(ticks[-1] - ticks[0]) * (ticks[-1] - ticks[-2]) <= W:
                        if self.specialTickClear:
                            labels[-2] = ''
                        else:
                            del ticks[-2], labels[-2]
                try:
                    if labels[0] and labels[0] == labels[1]:
                        del ticks[1], labels[1]
                except IndexError:
                    pass
                return (ticks, labels)
        raise ValueError('Problem selecting NormalDate value axis tick positions')

    def _convertXV(self, data):
        """Convert all XValues to a standard normalDate type"""
        VC = self._valueClass
        for D in data:
            for i in range(len(D)):
                x, y = D[i]
                if not isinstance(x, VC):
                    D[i] = (VC(x), y)

    def _getStepsAndLabels(self, xVals):
        if self.dailyFreq:
            xEOM = []
            pm = 0
            px = xVals[0]
            for x in xVals:
                m = x.month()
                if pm != m:
                    if pm:
                        xEOM.append(px)
                    pm = m
                px = x
            px = xVals[-1]
            if xEOM[-1] != x:
                xEOM.append(px)
            steps, labels = self._xAxisTicker(xEOM)
        else:
            steps, labels = self._xAxisTicker(xVals)
        return (steps, labels)

    def configure(self, data):
        self._convertXV(data)
        xVals = set()
        for x in data:
            for dv in x:
                xVals.add(dv[0])
        xVals = list(xVals)
        xVals.sort()
        VC = self._valueClass
        steps, labels = self._getStepsAndLabels(xVals)
        valueMin, valueMax = (self.valueMin, self.valueMax)
        valueMin = xVals[0] if valueMin is None else VC(valueMin)
        valueMax = xVals[-1] if valueMax is None else VC(valueMax)
        self._valueMin, self._valueMax = (valueMin, valueMax)
        self._tickValues = steps
        self._labelTextFormat = labels
        self._scaleFactor = self._length / float(valueMax - valueMin)
        self._tickValues = steps
        self._configured = 1
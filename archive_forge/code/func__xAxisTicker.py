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
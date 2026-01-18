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
def _calcPreliminaryWidths(self, availWidth):
    """Fallback algorithm for when main one fails.

        Where exact width info not given but things like
        paragraphs might be present, do a preliminary scan
        and assign some best-guess values."""
    W = list(self._argW)
    totalDefined = 0.0
    percentDefined = 0
    percentTotal = 0
    numberUndefined = 0
    numberGreedyUndefined = 0
    for w in W:
        if w is None:
            numberUndefined += 1
        elif w == '*':
            numberUndefined += 1
            numberGreedyUndefined += 1
        elif _endswith(w, '%'):
            percentDefined += 1
            percentTotal += float(w[:-1])
        else:
            assert isinstance(w, (int, float))
            totalDefined = totalDefined + w
    given = []
    sizeable = []
    unsizeable = []
    minimums = {}
    totalMinimum = 0
    elementWidth = self._elementWidth
    for colNo in range(self._ncols):
        w = W[colNo]
        if w is None or w == '*' or _endswith(w, '%'):
            siz = 1
            final = 0
            for rowNo in range(self._nrows):
                value = self._cellvalues[rowNo][colNo]
                style = self._cellStyles[rowNo][colNo]
                new = elementWidth(value, style) or 0
                new += style.leftPadding + style.rightPadding
                final = max(final, new)
                siz = siz and self._canGetWidth(value)
            if siz:
                sizeable.append(colNo)
            else:
                unsizeable.append(colNo)
            minimums[colNo] = final
            totalMinimum += final
        else:
            given.append(colNo)
    if len(given) == self._ncols:
        return
    remaining = availWidth - (totalMinimum + totalDefined)
    if remaining > 0:
        definedPercentage = totalDefined / float(availWidth) * 100
        percentTotal += definedPercentage
        if numberUndefined and percentTotal < 100:
            undefined = numberGreedyUndefined or numberUndefined
            defaultWeight = (100 - percentTotal) / float(undefined)
            percentTotal = 100
            defaultDesired = defaultWeight / float(percentTotal) * availWidth
        else:
            defaultWeight = defaultDesired = 1
        desiredWidths = []
        totalDesired = 0
        effectiveRemaining = remaining
        for colNo, minimum in minimums.items():
            w = W[colNo]
            if _endswith(w, '%'):
                desired = float(w[:-1]) / percentTotal * availWidth
            elif w == '*':
                desired = defaultDesired
            else:
                desired = not numberGreedyUndefined and defaultDesired or 1
            if desired <= minimum:
                W[colNo] = minimum
            else:
                desiredWidths.append((desired - minimum, minimum, desired, colNo))
                totalDesired += desired
                effectiveRemaining += minimum
        if desiredWidths:
            proportion = effectiveRemaining / float(totalDesired)
            desiredWidths.sort()
            finalSet = []
            for disappointment, minimum, desired, colNo in desiredWidths:
                adjusted = proportion * desired
                if adjusted < minimum:
                    W[colNo] = minimum
                    totalDesired -= desired
                    effectiveRemaining -= minimum
                    if totalDesired:
                        proportion = effectiveRemaining / float(totalDesired)
                else:
                    finalSet.append((minimum, desired, colNo))
            for minimum, desired, colNo in finalSet:
                adjusted = proportion * desired
                assert adjusted >= minimum
                W[colNo] = adjusted
    else:
        if percentTotal > 0:
            d = []
            for colNo, w in minimums.items():
                if w.endswith('%'):
                    W[colNo] = w = availWidth * float(w[:-1]) / percentTotal
                    totalDefined += w
                    d.append(colNo)
            for colNo in d:
                del minimums[colNo]
            del d
            totalMinimum = sum(minimums.values())
            remaining = availWidth - (totalDefined + totalMinimum)
        if remaining < 0 and totalDefined * rl_config.defCWRF + remaining >= 0:
            adj = -remaining / totalDefined
            for colNo, w in enumerate(W):
                if colNo not in minimums:
                    dw = adj * w
                    W[colNo] -= dw
                    totalDefined -= dw
            remaining = availWidth - (totalDefined + totalMinimum)
            adj = 1
        else:
            remaining = availWidth - totalDefined
            adj = 1 if remaining <= 0 else remaining / totalMinimum
        for colNo, minimum in minimums.items():
            W[colNo] = minimum * adj
    self._argW = self._colWidths = W
    return W
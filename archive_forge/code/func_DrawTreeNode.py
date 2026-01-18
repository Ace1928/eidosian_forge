import math
from rdkit.sping import pid as piddle
def DrawTreeNode(node, loc, canvas, nRes=2, scaleLeaves=False, showPurity=False):
    """Recursively displays the given tree node and all its children on the canvas
  """
    try:
        nChildren = node.totNChildren
    except AttributeError:
        nChildren = None
    if nChildren is None:
        CalcTreeNodeSizes(node)
    if not scaleLeaves or not node.GetTerminal():
        rad = visOpts.circRad
    else:
        scaleLoc = getattr(node, '_scaleLoc', 0.5)
        rad = visOpts.minCircRad + node._scaleLoc * (visOpts.maxCircRad - visOpts.minCircRad)
    x1 = loc[0] - rad
    y1 = loc[1] - rad
    x2 = loc[0] + rad
    y2 = loc[1] + rad
    if showPurity and node.GetTerminal():
        examples = node.GetExamples()
        nEx = len(examples)
        if nEx:
            tgtVal = int(node.GetLabel())
            purity = 0.0
            for ex in examples:
                if int(ex[-1]) == tgtVal:
                    purity += 1.0 / len(examples)
        else:
            purity = 1.0
        deg = purity * math.pi
        xFact = rad * math.sin(deg)
        yFact = rad * math.cos(deg)
        pureX = loc[0] + xFact
        pureY = loc[1] + yFact
    children = node.GetChildren()
    childY = loc[1] + visOpts.vertOffset
    childX = loc[0] - (visOpts.horizOffset + visOpts.circRad) * node.totNChildren / 2
    for i in range(len(children)):
        child = children[i]
        halfWidth = (visOpts.horizOffset + visOpts.circRad) * child.totNChildren / 2
        childX = childX + halfWidth
        childLoc = [childX, childY]
        canvas.drawLine(loc[0], loc[1], childLoc[0], childLoc[1], visOpts.lineColor, visOpts.lineWidth)
        DrawTreeNode(child, childLoc, canvas, nRes=nRes, scaleLeaves=scaleLeaves, showPurity=showPurity)
        childX = childX + halfWidth
    if node.GetTerminal():
        lab = node.GetLabel()
        cFac = float(lab) / float(nRes - 1)
        if hasattr(node, 'GetExamples') and node.GetExamples():
            theColor = (1.0 - cFac) * visOpts.terminalOffColor + cFac * visOpts.terminalOnColor
            outlColor = visOpts.outlineColor
        else:
            theColor = (1.0 - cFac) * visOpts.terminalOffColor + cFac * visOpts.terminalOnColor
            outlColor = visOpts.terminalEmptyColor
        canvas.drawEllipse(x1, y1, x2, y2, outlColor, visOpts.lineWidth, theColor)
        if showPurity:
            canvas.drawLine(loc[0], loc[1], pureX, pureY, piddle.Color(1, 1, 1), 2)
    else:
        theColor = visOpts.circColor
        canvas.drawEllipse(x1, y1, x2, y2, visOpts.outlineColor, visOpts.lineWidth, theColor)
        canvas.defaultFont = visOpts.labelFont
        labelStr = str(node.GetLabel())
        strLoc = (loc[0] - canvas.stringWidth(labelStr) / 2, loc[1] + canvas.fontHeight() / 4)
        canvas.drawString(labelStr, strLoc[0], strLoc[1])
    node._bBox = (x1, y1, x2, y2)
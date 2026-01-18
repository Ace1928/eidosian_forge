import numpy
from . import ClusterUtils
def _DrawToLimit(self, cluster):
    """
      we assume that _drawPos settings have been done already
    """
    if self.lineWidth is None:
        lineWidth = VisOpts.lineWidth
    else:
        lineWidth = self.lineWidth
    examine = [cluster]
    while len(examine):
        node = examine.pop(0)
        xp, yp = node._drawPos
        children = node.GetChildren()
        if abs(children[1]._drawPos[0] - children[0]._drawPos[0]) > self.tooClose:
            drawColor = VisOpts.lineColor
            self.canvas.drawLine(children[0]._drawPos[0], yp, children[-1]._drawPos[0], yp, drawColor, lineWidth)
            for child in children:
                if self.ptColors and child.GetData() is not None:
                    drawColor = self.ptColors[child.GetData()]
                else:
                    drawColor = VisOpts.lineColor
                cxp, cyp = child._drawPos
                self.canvas.drawLine(cxp, yp, cxp, cyp, drawColor, lineWidth)
                if not child.IsTerminal():
                    examine.append(child)
                elif self.showIndices and (not self.stopAtCentroids):
                    try:
                        txt = str(child.GetName())
                    except Exception:
                        txt = str(child.GetIndex())
                    self.canvas.drawString(txt, cxp - self.canvas.stringWidth(txt) / 2, cyp)
        else:
            self.canvas.drawLine(xp, yp, xp, self.size[1] - VisOpts.yOffset, VisOpts.hideColor, lineWidth)
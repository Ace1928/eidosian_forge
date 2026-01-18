import numpy
from . import ClusterUtils
def _DrawClusterTree(cluster, canvas, size, ptColors=[], lineWidth=None, showIndices=0, showNodes=1, stopAtCentroids=0, logScale=0, tooClose=-1):
    """ handles the work of drawing a cluster tree on a Sping canvas

    **Arguments**

      - cluster: the cluster tree to be drawn

      - canvas:  the Sping canvas on which to draw

      - size: the size of _canvas_

      - ptColors: if this is specified, the _colors_ will be used to color
        the terminal nodes of the cluster tree.  (color == _pid.Color_)

      - lineWidth: if specified, it will be used for the widths of the lines
        used to draw the tree

   **Notes**

     - _Canvas_ is neither _save_d nor _flush_ed at the end of this

     - if _ptColors_ is the wrong length for the number of possible terminal
       node types, this will throw an IndexError

     - terminal node types are determined using their _GetData()_ methods

  """
    if lineWidth is None:
        lineWidth = VisOpts.lineWidth
    pts = cluster.GetPoints()
    nPts = len(pts)
    if nPts <= 1:
        return
    xSpace = float(size[0] - 2 * VisOpts.xOffset) / float(nPts - 1)
    if logScale > 0:
        v = _scaleMetric(cluster.GetMetric(), logScale)
    else:
        v = float(cluster.GetMetric())
    ySpace = float(size[1] - 2 * VisOpts.yOffset) / v
    for i in range(nPts):
        pt = pts[i]
        if logScale > 0:
            v = _scaleMetric(pt.GetMetric(), logScale)
        else:
            v = float(pt.GetMetric())
        pt._drawPos = (VisOpts.xOffset + i * xSpace, size[1] - (v * ySpace + VisOpts.yOffset))
    if not stopAtCentroids:
        allNodes = ClusterUtils.GetNodeList(cluster)
    else:
        allNodes = ClusterUtils.GetNodesDownToCentroids(cluster)
    while len(allNodes):
        node = allNodes.pop(0)
        children = node.GetChildren()
        if len(children):
            if logScale > 0:
                v = _scaleMetric(node.GetMetric(), logScale)
            else:
                v = float(node.GetMetric())
            yp = size[1] - (v * ySpace + VisOpts.yOffset)
            childLocs = [x._drawPos[0] for x in children]
            xp = sum(childLocs) / float(len(childLocs))
            node._drawPos = (xp, yp)
            if not stopAtCentroids or node._aboveCentroid > 0:
                for child in children:
                    if ptColors != [] and child.GetData() is not None:
                        drawColor = ptColors[child.GetData()]
                    else:
                        drawColor = VisOpts.lineColor
                    if showNodes and hasattr(child, '_isCentroid'):
                        canvas.drawLine(child._drawPos[0], child._drawPos[1] - VisOpts.nodeRad / 2, child._drawPos[0], node._drawPos[1], drawColor, lineWidth)
                    else:
                        canvas.drawLine(child._drawPos[0], child._drawPos[1], child._drawPos[0], node._drawPos[1], drawColor, lineWidth)
                canvas.drawLine(children[0]._drawPos[0], node._drawPos[1], children[-1]._drawPos[0], node._drawPos[1], VisOpts.lineColor, lineWidth)
            else:
                for child in children:
                    drawColor = VisOpts.hideColor
                    canvas.drawLine(child._drawPos[0], child._drawPos[1], child._drawPos[0], node._drawPos[1], drawColor, VisOpts.hideWidth)
                canvas.drawLine(children[0]._drawPos[0], node._drawPos[1], children[-1]._drawPos[0], node._drawPos[1], VisOpts.hideColor, VisOpts.hideWidth)
        if showIndices and (not stopAtCentroids or node._aboveCentroid >= 0):
            txt = str(node.GetIndex())
            if hasattr(node, '_isCentroid'):
                txtColor = piddle.Color(1, 0.2, 0.2)
            else:
                txtColor = piddle.Color(0, 0, 0)
            canvas.drawString(txt, node._drawPos[0] - canvas.stringWidth(txt) / 2, node._drawPos[1] + canvas.fontHeight() / 4, color=txtColor)
        if showNodes and hasattr(node, '_isCentroid'):
            rad = VisOpts.nodeRad
            canvas.drawEllipse(node._drawPos[0] - rad / 2, node._drawPos[1] - rad / 2, node._drawPos[0] + rad / 2, node._drawPos[1] + rad / 2, piddle.transparent, fillColor=VisOpts.nodeColor)
            txt = str(node._clustID)
            canvas.drawString(txt, node._drawPos[0] - canvas.stringWidth(txt) / 2, node._drawPos[1] + canvas.fontHeight() / 4, color=piddle.Color(0, 0, 0))
    if showIndices and (not stopAtCentroids):
        for pt in pts:
            txt = str(pt.GetIndex())
            canvas.drawString(str(pt.GetIndex()), pt._drawPos[0] - canvas.stringWidth(txt) / 2, pt._drawPos[1])
from sping import pid as piddle
def DrawHierarchy(adjList, levelList, canvas, entryColors=None, bitIds=None, minLevel=-1, maxLevel=100000000.0):
    """

  Arguments:

   - adjList: adjacency list representation of the hierarchy to be drawn

   - levelList: dictionary mapping level -> list of ids

  """
    if bitIds is None:
        bitIds = []
    if entryColors is None:
        entryColors = {}
    levelLengths = levelList.keys()
    levelLengths.sort()
    minLevel = max(minLevel, levelLengths[0])
    maxLevel = min(maxLevel, levelLengths[-1])
    dims = canvas.size
    drawLocs = {}
    for levelLen in range(maxLevel, minLevel - 1, -1):
        nLevelsDown = levelLen - minLevel
        pos = [0, visOpts.vertOffset * nLevelsDown + visOpts.topMargin]
        ids = levelList.get(levelLen, [])
        nHere = len(ids)
        canvas.defaultFont = visOpts.labelFont
        if nHere:
            spacePerNode = float(dims[0]) / nHere
            spacePerNode -= visOpts.horizOffset
            nodeRad = max(spacePerNode / 2, visOpts.minCircRad)
            nodeRad = min(nodeRad, visOpts.maxCircRad)
            spacePerNode = nodeRad * 2 + visOpts.horizOffset
            pos[0] = dims[0] / 2.0
            if nHere % 2:
                pos[0] -= spacePerNode / 2
            pos[0] -= (nHere // 2 - 0.5) * spacePerNode
            for ID in ids:
                if not bitIds or ID in bitIds:
                    if levelLen != maxLevel:
                        for neighbor in adjList[ID]:
                            if neighbor in drawLocs:
                                p2 = drawLocs[neighbor][0]
                                canvas.drawLine(pos[0], pos[1], p2[0], p2[1], visOpts.lineColor, visOpts.lineWidth)
                    drawLocs[ID] = (tuple(pos), nodeRad)
                    pos[0] += spacePerNode
    for ID in drawLocs.keys():
        pos, nodeRad = drawLocs[ID]
        x1, y1 = (pos[0] - nodeRad, pos[1] - nodeRad)
        x2, y2 = (pos[0] + nodeRad, pos[1] + nodeRad)
        drawColor = entryColors.get(ID, visOpts.circColor)
        canvas.drawEllipse(x1, y1, x2, y2, visOpts.outlineColor, 0, drawColor)
        label = str(ID)
        txtLoc = (pos[0] + canvas.fontHeight() / 4, pos[1] + canvas.stringWidth(label) / 2)
        canvas.drawString(label, txtLoc[0], txtLoc[1], angle=90)
    return drawLocs
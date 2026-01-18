import math
def _getLinePoints(self, p1, p2, dash):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    lineLen = math.sqrt(dx * dx + dy * dy)
    theta = math.atan2(dy, dx)
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    pos = (x1, y1)
    pts = [pos]
    dist = 0
    currDash = 0
    while dist < lineLen:
        currL = dash[currDash % len(dash)]
        if dist + currL > lineLen:
            currL = lineLen - dist
        endP = (pos[0] + currL * cosT, pos[1] + currL * sinT)
        pts.append(endP)
        pos = endP
        dist += currL
        currDash += 1
    return pts
import math
from rdkit.sping import pid as piddle
class VisOpts(object):
    circRad = 10
    minCircRad = 4
    maxCircRad = 16
    circColor = piddle.Color(0.6, 0.6, 0.9)
    terminalEmptyColor = piddle.Color(0.8, 0.8, 0.2)
    terminalOnColor = piddle.Color(0.8, 0.8, 0.8)
    terminalOffColor = piddle.Color(0.2, 0.2, 0.2)
    outlineColor = piddle.transparent
    lineColor = piddle.Color(0, 0, 0)
    lineWidth = 2
    horizOffset = 10
    vertOffset = 50
    labelFont = piddle.Font(face='helvetica', size=10)
    highlightColor = piddle.Color(1.0, 1.0, 0.4)
    highlightWidth = 2
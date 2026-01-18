import math
from rdkit.sping import pid as piddle
def _ApplyNodeScales(node, min, max):
    if node.GetTerminal():
        if max != min:
            loc = float(node.nExamples - min) / (max - min)
        else:
            loc = 0.5
        node._scaleLoc = loc
    else:
        for child in node.GetChildren():
            _ApplyNodeScales(child, min, max)
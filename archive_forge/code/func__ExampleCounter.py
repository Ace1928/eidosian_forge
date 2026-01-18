import math
from rdkit.sping import pid as piddle
def _ExampleCounter(node, min, max):
    if node.GetTerminal():
        cnt = node.nExamples
        if cnt < min:
            min = cnt
        if cnt > max:
            max = cnt
    else:
        for child in node.GetChildren():
            provMin, provMax = _ExampleCounter(child, min, max)
            if provMin < min:
                min = provMin
            if provMax > max:
                max = provMax
    return (min, max)
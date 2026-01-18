import math
from rdkit.sping import pid as piddle
def SetNodeScales(node):
    min, max = (100000000.0, -100000000.0)
    min, max = _ExampleCounter(node, min, max)
    node._scales = (min, max)
    _ApplyNodeScales(node, min, max)
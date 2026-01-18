import math
from rdkit.sping import pid as piddle
def _simpleTest(canv):
    from .Tree import TreeNode as Node
    root = Node(None, 'r', label='r')
    c1 = root.AddChild('l1_1', label='l1_1')
    c2 = root.AddChild('l1_2', isTerminal=1, label=1)
    c3 = c1.AddChild('l2_1', isTerminal=1, label=0)
    c4 = c1.AddChild('l2_2', isTerminal=1, label=1)
    DrawTreeNode(root, (150, visOpts.vertOffset), canv)
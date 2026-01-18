import random
import sys
from . import Nodes
def _get_branches(node):
    branches = []
    for b in self.node(node).succ:
        branches.append([node, b, self.node(b).data.branchlength, self.node(b).data.support])
        branches.extend(_get_branches(b))
    return branches
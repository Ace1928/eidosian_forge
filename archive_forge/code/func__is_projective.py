import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def _is_projective(self, depgraph):
    arc_list = []
    for key in depgraph.nodes:
        node = depgraph.nodes[key]
        if 'head' in node:
            childIdx = node['address']
            parentIdx = node['head']
            if parentIdx is not None:
                arc_list.append((parentIdx, childIdx))
    for parentIdx, childIdx in arc_list:
        if childIdx > parentIdx:
            temp = childIdx
            childIdx = parentIdx
            parentIdx = temp
        for k in range(childIdx + 1, parentIdx):
            for m in range(len(depgraph.nodes)):
                if m < childIdx or m > parentIdx:
                    if (k, m) in arc_list:
                        return False
                    if (m, k) in arc_list:
                        return False
    return True
import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def _make_species(self):
    a = tree.Node('animal')
    m = tree.Node('mammal')
    r = tree.Node('reptile')
    a.add(m)
    a.add(r)
    m.add(tree.Node('horse'))
    p = tree.Node('primate')
    m.add(p)
    p.add(tree.Node('monkey'))
    p.add(tree.Node('human'))
    return a
import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def G1():
    G = nx.from_numpy_array(G_array, create_using=nx.MultiDiGraph)
    return G
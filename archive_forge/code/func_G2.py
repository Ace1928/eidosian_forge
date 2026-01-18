import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def G2():
    Garr = G_array.copy()
    Garr[np.nonzero(Garr)] -= 10
    G = nx.from_numpy_array(Garr, create_using=nx.MultiDiGraph)
    return G
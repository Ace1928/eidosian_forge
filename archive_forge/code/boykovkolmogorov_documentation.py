from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
Adoption stage.

        Reconstruct search trees by adopting or discarding orphans.
        During augmentation stage some edges got saturated and thus
        the source and target search trees broke down to forests, with
        orphans as roots of some of its trees. We have to reconstruct
        the search trees rooted to source and target before we can grow
        them again.
        
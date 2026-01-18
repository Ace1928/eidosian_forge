from collections import deque
import networkx as nx
class Level:
    """Active and inactive nodes in a level."""
    __slots__ = ('active', 'inactive')

    def __init__(self):
        self.active = set()
        self.inactive = set()
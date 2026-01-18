import networkx as nx
from collections import deque
def DAG(self):
    """
        Return the directed acyclic graph whose vertices are the
        strong components of the underlying digraph.  There is an edge
        joining two components if and only if there is an edge of the
        underlying digraph having an endpoint in each component.
        Using self.links, rather than self.digraph.edges, is a slight
        optimization.
        """
    edges = set()
    for tail, head in self.links:
        dag_tail = self.which_component[tail]
        dag_head = self.which_component[head]
        if dag_head != dag_tail:
            edges.add((dag_tail, dag_head))
    return Digraph(edges, self.components)
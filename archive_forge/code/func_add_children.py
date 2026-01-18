from itertools import chain
import networkx as nx
def add_children(parent, children_):
    for data in children_:
        child = data[ident]
        graph.add_edge(parent, child)
        grandchildren = data.get(children, [])
        if grandchildren:
            add_children(child, grandchildren)
        nodedata = {str(k): v for k, v in data.items() if k != ident and k != children}
        graph.add_node(child, **nodedata)
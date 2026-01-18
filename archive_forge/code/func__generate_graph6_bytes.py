from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
def _generate_graph6_bytes(G, nodes, header):
    """Yield bytes in the graph6 encoding of a graph.

    `G` is an undirected simple graph. `nodes` is the list of nodes for
    which the node-induced subgraph will be encoded; if `nodes` is the
    list of all nodes in the graph, the entire graph will be
    encoded. `header` is a Boolean that specifies whether to generate
    the header ``b'>>graph6<<'`` before the remaining data.

    This function generates `bytes` objects in the following order:

    1. the header (if requested),
    2. the encoding of the number of nodes,
    3. each character, one-at-a-time, in the encoding of the requested
       node-induced subgraph,
    4. a newline character.

    This function raises :exc:`ValueError` if the graph is too large for
    the graph6 format (that is, greater than ``2 ** 36`` nodes).

    """
    n = len(G)
    if n >= 2 ** 36:
        raise ValueError('graph6 is only defined if number of nodes is less than 2 ** 36')
    if header:
        yield b'>>graph6<<'
    for d in n_to_data(n):
        yield str.encode(chr(d + 63))
    bits = (nodes[j] in G[nodes[i]] for j in range(1, n) for i in range(j))
    chunk = list(islice(bits, 6))
    while chunk:
        d = sum((b << 5 - i for i, b in enumerate(chunk)))
        yield str.encode(chr(d + 63))
        chunk = list(islice(bits, 6))
    yield b'\n'
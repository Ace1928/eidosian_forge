import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
def _generate_sparse6_bytes(G, nodes, header):
    """Yield bytes in the sparse6 encoding of a graph.

    `G` is an undirected simple graph. `nodes` is the list of nodes for
    which the node-induced subgraph will be encoded; if `nodes` is the
    list of all nodes in the graph, the entire graph will be
    encoded. `header` is a Boolean that specifies whether to generate
    the header ``b'>>sparse6<<'`` before the remaining data.

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
        raise ValueError('sparse6 is only defined if number of nodes is less than 2 ** 36')
    if header:
        yield b'>>sparse6<<'
    yield b':'
    for d in n_to_data(n):
        yield str.encode(chr(d + 63))
    k = 1
    while 1 << k < n:
        k += 1

    def enc(x):
        """Big endian k-bit encoding of x"""
        return [1 if x & 1 << k - 1 - i else 0 for i in range(k)]
    edges = sorted(((max(u, v), min(u, v)) for u, v in G.edges()))
    bits = []
    curv = 0
    for v, u in edges:
        if v == curv:
            bits.append(0)
            bits.extend(enc(u))
        elif v == curv + 1:
            curv += 1
            bits.append(1)
            bits.extend(enc(u))
        else:
            curv = v
            bits.append(1)
            bits.extend(enc(v))
            bits.append(0)
            bits.extend(enc(u))
    if k < 6 and n == 1 << k and (-len(bits) % 6 >= k) and (curv < n - 1):
        bits.append(0)
        bits.extend([1] * (-len(bits) % 6))
    else:
        bits.extend([1] * (-len(bits) % 6))
    data = [(bits[i + 0] << 5) + (bits[i + 1] << 4) + (bits[i + 2] << 3) + (bits[i + 3] << 2) + (bits[i + 4] << 1) + (bits[i + 5] << 0) for i in range(0, len(bits), 6)]
    for d in data:
        yield str.encode(chr(d + 63))
    yield b'\n'
import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
class EdgeComponentAuxGraph:
    """A simple algorithm to find all k-edge-connected components in a graph.

    Constructing the auxiliary graph (which may take some time) allows for the
    k-edge-ccs to be found in linear time for arbitrary k.

    Notes
    -----
    This implementation is based on [1]_. The idea is to construct an auxiliary
    graph from which the k-edge-ccs can be extracted in linear time. The
    auxiliary graph is constructed in $O(|V|\\cdot F)$ operations, where F is the
    complexity of max flow. Querying the components takes an additional $O(|V|)$
    operations. This algorithm can be slow for large graphs, but it handles an
    arbitrary k and works for both directed and undirected inputs.

    The undirected case for k=1 is exactly connected components.
    The undirected case for k=2 is exactly bridge connected components.
    The directed case for k=1 is exactly strongly connected components.

    References
    ----------
    .. [1] Wang, Tianhao, et al. (2015) A simple algorithm for finding all
        k-edge-connected components.
        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136264

    Examples
    --------
    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> from networkx.algorithms.connectivity import EdgeComponentAuxGraph
    >>> # Build an interesting graph with multiple levels of k-edge-ccs
    >>> paths = [
    ...     (1, 2, 3, 4, 1, 3, 4, 2),  # a 3-edge-cc (a 4 clique)
    ...     (5, 6, 7, 5),  # a 2-edge-cc (a 3 clique)
    ...     (1, 5),  # combine first two ccs into a 1-edge-cc
    ...     (0,),  # add an additional disconnected 1-edge-cc
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> # Constructing the AuxGraph takes about O(n ** 4)
    >>> aux_graph = EdgeComponentAuxGraph.construct(G)
    >>> # Once constructed, querying takes O(n)
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=1)))
    [[0], [1, 2, 3, 4, 5, 6, 7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=2)))
    [[0], [1, 2, 3, 4], [5, 6, 7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=3)))
    [[0], [1, 2, 3, 4], [5], [6], [7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=4)))
    [[0], [1], [2], [3], [4], [5], [6], [7]]

    The auxiliary graph is primarily used for k-edge-ccs but it
    can also speed up the queries of k-edge-subgraphs by refining the
    search space.

    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> from networkx.algorithms.connectivity import EdgeComponentAuxGraph
    >>> paths = [
    ...     (1, 2, 4, 3, 1, 4),
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> aux_graph = EdgeComponentAuxGraph.construct(G)
    >>> sorted(map(sorted, aux_graph.k_edge_subgraphs(k=3)))
    [[1], [2], [3], [4]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=3)))
    [[1, 4], [2], [3]]
    """

    @classmethod
    def construct(EdgeComponentAuxGraph, G):
        """Builds an auxiliary graph encoding edge-connectivity between nodes.

        Notes
        -----
        Given G=(V, E), initialize an empty auxiliary graph A.
        Choose an arbitrary source node s.  Initialize a set N of available
        nodes (that can be used as the sink). The algorithm picks an
        arbitrary node t from N - {s}, and then computes the minimum st-cut
        (S, T) with value w. If G is directed the minimum of the st-cut or
        the ts-cut is used instead. Then, the edge (s, t) is added to the
        auxiliary graph with weight w. The algorithm is called recursively
        first using S as the available nodes and s as the source, and then
        using T and t. Recursion stops when the source is the only available
        node.

        Parameters
        ----------
        G : NetworkX graph
        """
        not_implemented_for('multigraph')(lambda G: G)(G)

        def _recursive_build(H, A, source, avail):
            if {source} == avail:
                return
            sink = arbitrary_element(avail - {source})
            value, (S, T) = nx.minimum_cut(H, source, sink)
            if H.is_directed():
                value_, (T_, S_) = nx.minimum_cut(H, sink, source)
                if value_ < value:
                    value, S, T = (value_, S_, T_)
            A.add_edge(source, sink, weight=value)
            _recursive_build(H, A, source, avail.intersection(S))
            _recursive_build(H, A, sink, avail.intersection(T))
        H = G.__class__()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(), capacity=1)
        A = nx.Graph()
        if H.number_of_nodes() > 0:
            source = arbitrary_element(H.nodes())
            avail = set(H.nodes())
            _recursive_build(H, A, source, avail)
        self = EdgeComponentAuxGraph()
        self.A = A
        self.H = H
        return self

    def k_edge_components(self, k):
        """Queries the auxiliary graph for k-edge-connected components.

        Parameters
        ----------
        k : Integer
            Desired edge connectivity

        Returns
        -------
        k_edge_components : a generator of k-edge-ccs

        Notes
        -----
        Given the auxiliary graph, the k-edge-connected components can be
        determined in linear time by removing all edges with weights less than
        k from the auxiliary graph.  The resulting connected components are the
        k-edge-ccs in the original graph.
        """
        if k < 1:
            raise ValueError('k cannot be less than 1')
        A = self.A
        aux_weights = nx.get_edge_attributes(A, 'weight')
        R = nx.Graph()
        R.add_nodes_from(A.nodes())
        R.add_edges_from((e for e, w in aux_weights.items() if w >= k))
        yield from nx.connected_components(R)

    def k_edge_subgraphs(self, k):
        """Queries the auxiliary graph for k-edge-connected subgraphs.

        Parameters
        ----------
        k : Integer
            Desired edge connectivity

        Returns
        -------
        k_edge_subgraphs : a generator of k-edge-subgraphs

        Notes
        -----
        Refines the k-edge-ccs into k-edge-subgraphs. The running time is more
        than $O(|V|)$.

        For single values of k it is faster to use `nx.k_edge_subgraphs`.
        But for multiple values of k, it can be faster to build AuxGraph and
        then use this method.
        """
        if k < 1:
            raise ValueError('k cannot be less than 1')
        H = self.H
        A = self.A
        aux_weights = nx.get_edge_attributes(A, 'weight')
        R = nx.Graph()
        R.add_nodes_from(A.nodes())
        R.add_edges_from((e for e, w in aux_weights.items() if w >= k))
        for cc in nx.connected_components(R):
            if len(cc) < k:
                for node in cc:
                    yield {node}
            else:
                C = H.subgraph(cc)
                yield from k_edge_subgraphs(C, k)
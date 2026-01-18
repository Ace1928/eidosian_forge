import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
class RAG(nx.Graph):
    """The Region Adjacency Graph (RAG) of an image, subclasses :obj:`networkx.Graph`.

    Parameters
    ----------
    label_image : array of int
        An initial segmentation, with each region labeled as a different
        integer. Every unique value in ``label_image`` will correspond to
        a node in the graph.
    connectivity : int in {1, ..., ``label_image.ndim``}, optional
        The connectivity between pixels in ``label_image``. For a 2D image,
        a connectivity of 1 corresponds to immediate neighbors up, down,
        left, and right, while a connectivity of 2 also includes diagonal
        neighbors. See :func:`scipy.ndimage.generate_binary_structure`.
    data : :obj:`networkx.Graph` specification, optional
        Initial or additional edges to pass to :obj:`networkx.Graph`
        constructor. Valid edge specifications include edge list (list of tuples),
        NumPy arrays, and SciPy sparse matrices.
    **attr : keyword arguments, optional
        Additional attributes to add to the graph.
    """

    def __init__(self, label_image=None, connectivity=1, data=None, **attr):
        super().__init__(data, **attr)
        if self.number_of_nodes() == 0:
            self.max_id = 0
        else:
            self.max_id = max(self.nodes())
        if label_image is not None:
            fp = ndi.generate_binary_structure(label_image.ndim, connectivity)
            output = np.broadcast_to(1.0, label_image.shape)
            output.setflags(write=True)
            ndi.generic_filter(label_image, function=_add_edge_filter, footprint=fp, mode='nearest', output=output, extra_arguments=(self,))

    def merge_nodes(self, src, dst, weight_func=min_weight, in_place=True, extra_arguments=None, extra_keywords=None):
        """Merge node `src` and `dst`.

        The new combined node is adjacent to all the neighbors of `src`
        and `dst`. `weight_func` is called to decide the weight of edges
        incident on the new node.

        Parameters
        ----------
        src, dst : int
            Nodes to be merged.
        weight_func : callable, optional
            Function to decide the attributes of edges incident on the new
            node. For each neighbor `n` for `src` and `dst`, `weight_func` will
            be called as follows: `weight_func(src, dst, n, *extra_arguments,
            **extra_keywords)`. `src`, `dst` and `n` are IDs of vertices in the
            RAG object which is in turn a subclass of :obj:`networkx.Graph`. It is
            expected to return a dict of attributes of the resulting edge.
        in_place : bool, optional
            If set to `True`, the merged node has the id `dst`, else merged
            node has a new id which is returned.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords : dictionary, optional
            The dict of keyword arguments passed to the `weight_func`.

        Returns
        -------
        id : int
            The id of the new node.

        Notes
        -----
        If `in_place` is `False` the resulting node has a new id, rather than
        `dst`.
        """
        if extra_arguments is None:
            extra_arguments = []
        if extra_keywords is None:
            extra_keywords = {}
        src_nbrs = set(self.neighbors(src))
        dst_nbrs = set(self.neighbors(dst))
        neighbors = (src_nbrs | dst_nbrs) - {src, dst}
        if in_place:
            new = dst
        else:
            new = self.next_id()
            self.add_node(new)
        for neighbor in neighbors:
            data = weight_func(self, src, dst, neighbor, *extra_arguments, **extra_keywords)
            self.add_edge(neighbor, new, attr_dict=data)
        self.nodes[new]['labels'] = self.nodes[src]['labels'] + self.nodes[dst]['labels']
        self.remove_node(src)
        if not in_place:
            self.remove_node(dst)
        return new

    def add_node(self, n, attr_dict=None, **attr):
        """Add node `n` while updating the maximum node id.

        .. seealso:: :obj:`networkx.Graph.add_node`."""
        if attr_dict is None:
            attr_dict = attr
        else:
            attr_dict.update(attr)
        super().add_node(n, **attr_dict)
        self.max_id = max(n, self.max_id)

    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add an edge between `u` and `v` while updating max node id.

        .. seealso:: :obj:`networkx.Graph.add_edge`."""
        if attr_dict is None:
            attr_dict = attr
        else:
            attr_dict.update(attr)
        super().add_edge(u, v, **attr_dict)
        self.max_id = max(u, v, self.max_id)

    def copy(self):
        """Copy the graph with its max node id.

        .. seealso:: :obj:`networkx.Graph.copy`."""
        g = super().copy()
        g.max_id = self.max_id
        return g

    def fresh_copy(self):
        """Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.

        This is required when subclassing Graph with networkx v2 and
        does not cause problems for v1. Here is more detail from
        the network migrating from 1.x to 2.x document::

            With the new GraphViews (SubGraph, ReversedGraph, etc)
            you can't assume that ``G.__class__()`` will create a new
            instance of the same graph type as ``G``. In fact, the
            call signature for ``__class__`` differs depending on
            whether ``G`` is a view or a base class. For v2.x you
            should use ``G.fresh_copy()`` to create a null graph of
            the correct type---ready to fill with nodes and edges.

        """
        return RAG()

    def next_id(self):
        """Returns the `id` for the new node to be inserted.

        The current implementation returns one more than the maximum `id`.

        Returns
        -------
        id : int
            The `id` of the new node to be inserted.
        """
        return self.max_id + 1

    def _add_node_silent(self, n):
        """Add node `n` without updating the maximum node id.

        This is a convenience method used internally.

        .. seealso:: :obj:`networkx.Graph.add_node`."""
        super().add_node(n)
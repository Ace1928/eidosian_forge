from collections import defaultdict
import networkx as nx
def add_half_edge(self, start_node, end_node, *, cw=None, ccw=None):
    """Adds a half-edge from `start_node` to `end_node`.

        If the half-edge is not the first one out of `start_node`, a reference
        node must be provided either in the clockwise (parameter `cw`) or in
        the counterclockwise (parameter `ccw`) direction. Only one of `cw`/`ccw`
        can be specified (or neither in the case of the first edge).
        Note that specifying a reference in the clockwise (`cw`) direction means
        inserting the new edge in the first counterclockwise position with
        respect to the reference (and vice-versa).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        cw, ccw: node
            End node of reference edge.
            Omit or pass `None` if adding the first out-half-edge of `start_node`.


        Raises
        ------
        NetworkXException
            If the `cw` or `ccw` node is not a successor of `start_node`.
            If `start_node` has successors, but neither `cw` or `ccw` is provided.
            If both `cw` and `ccw` are specified.

        See Also
        --------
        connect_components
        """
    succs = self._succ.get(start_node)
    if succs:
        leftmost_nbr = next(reversed(self._succ[start_node]))
        if cw is not None:
            if cw not in succs:
                raise nx.NetworkXError('Invalid clockwise reference node.')
            if ccw is not None:
                raise nx.NetworkXError('Only one of cw/ccw can be specified.')
            ref_ccw = succs[cw]['ccw']
            super().add_edge(start_node, end_node, cw=cw, ccw=ref_ccw)
            succs[ref_ccw]['cw'] = end_node
            succs[cw]['ccw'] = end_node
            move_leftmost_nbr_to_end = cw != leftmost_nbr
        elif ccw is not None:
            if ccw not in succs:
                raise nx.NetworkXError('Invalid counterclockwise reference node.')
            ref_cw = succs[ccw]['cw']
            super().add_edge(start_node, end_node, cw=ref_cw, ccw=ccw)
            succs[ref_cw]['ccw'] = end_node
            succs[ccw]['cw'] = end_node
            move_leftmost_nbr_to_end = True
        else:
            raise nx.NetworkXError('Node already has out-half-edge(s), either cw or ccw reference node required.')
        if move_leftmost_nbr_to_end:
            succs[leftmost_nbr] = succs.pop(leftmost_nbr)
    else:
        if cw is not None or ccw is not None:
            raise nx.NetworkXError('Invalid reference node.')
        super().add_edge(start_node, end_node, ccw=end_node, cw=end_node)
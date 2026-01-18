from collections import defaultdict
import networkx as nx
def add_half_edge_ccw(self, start_node, end_node, reference_neighbor):
    """Adds a half-edge from start_node to end_node.

        The half-edge is added counter clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge_cw
        connect_components
        add_half_edge_first

        """
    if reference_neighbor is None:
        self.add_edge(start_node, end_node)
        self[start_node][end_node]['cw'] = end_node
        self[start_node][end_node]['ccw'] = end_node
        self.nodes[start_node]['first_nbr'] = end_node
    else:
        ccw_reference = self[start_node][reference_neighbor]['ccw']
        self.add_half_edge_cw(start_node, end_node, ccw_reference)
        if reference_neighbor == self.nodes[start_node].get('first_nbr', None):
            self.nodes[start_node]['first_nbr'] = end_node
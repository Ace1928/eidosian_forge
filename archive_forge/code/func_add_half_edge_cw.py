from collections import defaultdict
import networkx as nx
def add_half_edge_cw(self, start_node, end_node, reference_neighbor):
    """Adds a half-edge from start_node to end_node.

        The half-edge is added clockwise next to the existing half-edge
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
        add_half_edge_ccw
        connect_components
        add_half_edge_first
        """
    self.add_edge(start_node, end_node)
    if reference_neighbor is None:
        self[start_node][end_node]['cw'] = end_node
        self[start_node][end_node]['ccw'] = end_node
        self.nodes[start_node]['first_nbr'] = end_node
        return
    if reference_neighbor not in self[start_node]:
        raise nx.NetworkXException('Cannot add edge. Reference neighbor does not exist')
    cw_reference = self[start_node][reference_neighbor]['cw']
    self[start_node][reference_neighbor]['cw'] = end_node
    self[start_node][end_node]['cw'] = cw_reference
    self[start_node][cw_reference]['ccw'] = end_node
    self[start_node][end_node]['ccw'] = reference_neighbor
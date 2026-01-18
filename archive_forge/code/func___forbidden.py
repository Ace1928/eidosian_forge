from collections import defaultdict
import networkx as nx
def __forbidden(self, *args, **kwargs):
    """Forbidden operation

        Any edge additions to a PlanarEmbedding should be done using
        method `add_half_edge`.
        """
    raise NotImplementedError('Use `add_half_edge` method to add edges to a PlanarEmbedding.')
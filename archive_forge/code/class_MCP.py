import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
class MCP(mcp.MCP_Connect):

    def _reset(self):
        """Reset the id map."""
        mcp.MCP_Connect._reset(self)
        self._conn = {}
        self._bestconn = {}

    def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
        hash = (min(id1, id2), max(id1, id2))
        val = (min(pos1, pos2), max(pos1, pos2))
        cost = min(cost1, cost2)
        self._conn.setdefault(hash, []).append(val)
        curcost = self._bestconn.get(hash, (np.inf,))[0]
        if cost < curcost:
            self._bestconn[hash] = (cost,) + val
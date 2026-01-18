from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _least_readv_node_readv(self, nodes):
    """Generate request groups for nodes using the least readv's.

        :param nodes: An iterable of graph index nodes.
        :return: Total node count and an iterator of the data needed to perform
            readvs to obtain the data for nodes. Each item yielded by the
            iterator is a tuple with:
            index, readv_vector, node_vector. readv_vector is a list ready to
            hand to the transport readv method, and node_vector is a list of
            (key, eol_flag, references) for the node retrieved by the
            matching readv_vector.
        """
    nodes = sorted(nodes)
    total = len(nodes)
    request_groups = {}
    for index, key, value, references in nodes:
        if index not in request_groups:
            request_groups[index] = []
        request_groups[index].append((key, value, references))
    result = []
    for index, items in request_groups.items():
        pack_readv_requests = []
        for key, value, references in items:
            bits = value[1:].split(b' ')
            offset, length = (int(bits[0]), int(bits[1]))
            pack_readv_requests.append(((offset, length), (key, value[0:1], references)))
        pack_readv_requests.sort()
        pack_readv = [readv for readv, node in pack_readv_requests]
        node_vector = [node for readv, node in pack_readv_requests]
        result.append((index, pack_readv, node_vector))
    return (total, result)
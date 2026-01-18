from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class ExtraParentsProvider:

    def get_parent_map(self, keys):
        return {b'rev1': [], b'rev2': [b'rev1']}
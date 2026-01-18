from io import BytesIO
from .. import errors, osutils, transport
from ..commands import Command, display_command
from ..option import Option
from ..workingtree import WorkingTree
from . import btree_index, static_tuple
def _dump_entries(self, trans, basename):
    try:
        st = trans.stat(basename)
    except errors.TransportNotPossible:
        bt, _ = self._get_index_and_bytes(trans, basename)
    else:
        bt = btree_index.BTreeGraphIndex(trans, basename, st.st_size)
    for node in bt.iter_all_entries():
        try:
            refs = node[3]
        except IndexError:
            refs_as_tuples = None
        else:
            refs_as_tuples = static_tuple.as_tuples(refs)
        if refs_as_tuples is not None:
            refs_as_tuples = tuple((tuple((tuple((r.decode('utf-8') for r in t1)) for t1 in t2)) for t2 in refs_as_tuples))
        as_tuple = (tuple([r.decode('utf-8') for r in node[1]]), node[2].decode('utf-8'), refs_as_tuples)
        self.outf.write('{}\n'.format(as_tuple))
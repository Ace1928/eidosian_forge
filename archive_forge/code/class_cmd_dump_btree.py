from io import BytesIO
from .. import errors, osutils, transport
from ..commands import Command, display_command
from ..option import Option
from ..workingtree import WorkingTree
from . import btree_index, static_tuple
class cmd_dump_btree(Command):
    __doc__ = 'Dump the contents of a btree index file to stdout.\n\n    PATH is a btree index file, it can be any URL. This includes things like\n    .bzr/repository/pack-names, or .bzr/repository/indices/a34b3a...ca4a4.iix\n\n    By default, the tuples stored in the index file will be displayed. With\n    --raw, we will uncompress the pages, but otherwise display the raw bytes\n    stored in the index.\n    '
    hidden = True
    encoding_type = 'exact'
    takes_args = ['path']
    takes_options = [Option('raw', help='Write the uncompressed bytes out, rather than the parsed tuples.')]

    def run(self, path, raw=False):
        dirname, basename = osutils.split(path)
        t = transport.get_transport(dirname)
        if raw:
            self._dump_raw_bytes(t, basename)
        else:
            self._dump_entries(t, basename)

    def _get_index_and_bytes(self, trans, basename):
        """Create a BTreeGraphIndex and raw bytes."""
        bt = btree_index.BTreeGraphIndex(trans, basename, None)
        bytes = trans.get_bytes(basename)
        bt._file = BytesIO(bytes)
        bt._size = len(bytes)
        return (bt, bytes)

    def _dump_raw_bytes(self, trans, basename):
        import zlib
        bt, bytes = self._get_index_and_bytes(trans, basename)
        for page_idx, page_start in enumerate(range(0, len(bytes), btree_index._PAGE_SIZE)):
            page_end = min(page_start + btree_index._PAGE_SIZE, len(bytes))
            page_bytes = bytes[page_start:page_end]
            if page_idx == 0:
                self.outf.write('Root node:\n')
                header_end, data = bt._parse_header_from_bytes(page_bytes)
                self.outf.write(page_bytes[:header_end])
                page_bytes = data
            self.outf.write('\nPage %d\n' % (page_idx,))
            if len(page_bytes) == 0:
                self.outf.write('(empty)\n')
            else:
                decomp_bytes = zlib.decompress(page_bytes)
                self.outf.write(decomp_bytes)
                self.outf.write('\n')

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
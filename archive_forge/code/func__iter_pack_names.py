import posixpath
import tempfile
from ..object_store import BucketBasedObjectStore
from ..pack import PACK_SPOOL_FILE_MAX_SIZE, Pack, PackData, load_pack_index_file
def _iter_pack_names(self):
    packs = {}
    for blob in self.bucket.list_blobs(prefix=self.subpath):
        name, ext = posixpath.splitext(posixpath.basename(blob.name))
        packs.setdefault(name, set()).add(ext)
    for name, exts in packs.items():
        if exts == {'.pack', '.idx'}:
            yield name
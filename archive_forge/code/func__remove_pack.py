import posixpath
import tempfile
from ..object_store import BucketBasedObjectStore
from ..pack import PACK_SPOOL_FILE_MAX_SIZE, Pack, PackData, load_pack_index_file
def _remove_pack(self, name):
    self.bucket.delete_blobs([posixpath.join(self.subpath, name) + '.' + ext for ext in ['pack', 'idx']])
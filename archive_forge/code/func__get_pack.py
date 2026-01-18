import posixpath
import tempfile
from ..object_store import BucketBasedObjectStore
from ..pack import PACK_SPOOL_FILE_MAX_SIZE, Pack, PackData, load_pack_index_file
def _get_pack(self, name):
    return Pack.from_lazy_objects(lambda: self._load_pack_data(name), lambda: self._load_pack_index(name))
import posixpath
import tempfile
from ..object_store import BucketBasedObjectStore
from ..pack import PACK_SPOOL_FILE_MAX_SIZE, Pack, PackData, load_pack_index_file
def _upload_pack(self, basename, pack_file, index_file):
    idxblob = self.bucket.blob(posixpath.join(self.subpath, basename + '.idx'))
    datablob = self.bucket.blob(posixpath.join(self.subpath, basename + '.pack'))
    idxblob.upload_from_file(index_file)
    datablob.upload_from_file(pack_file)
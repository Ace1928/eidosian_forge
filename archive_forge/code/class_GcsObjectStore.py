import posixpath
import tempfile
from ..object_store import BucketBasedObjectStore
from ..pack import PACK_SPOOL_FILE_MAX_SIZE, Pack, PackData, load_pack_index_file
class GcsObjectStore(BucketBasedObjectStore):

    def __init__(self, bucket, subpath='') -> None:
        super().__init__()
        self.bucket = bucket
        self.subpath = subpath

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.bucket!r}, subpath={self.subpath!r})'

    def _remove_pack(self, name):
        self.bucket.delete_blobs([posixpath.join(self.subpath, name) + '.' + ext for ext in ['pack', 'idx']])

    def _iter_pack_names(self):
        packs = {}
        for blob in self.bucket.list_blobs(prefix=self.subpath):
            name, ext = posixpath.splitext(posixpath.basename(blob.name))
            packs.setdefault(name, set()).add(ext)
        for name, exts in packs.items():
            if exts == {'.pack', '.idx'}:
                yield name

    def _load_pack_data(self, name):
        b = self.bucket.blob(posixpath.join(self.subpath, name + '.pack'))
        f = tempfile.SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE)
        b.download_to_file(f)
        f.seek(0)
        return PackData(name + '.pack', f)

    def _load_pack_index(self, name):
        b = self.bucket.blob(posixpath.join(self.subpath, name + '.idx'))
        f = tempfile.SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE)
        b.download_to_file(f)
        f.seek(0)
        return load_pack_index_file(name + '.idx', f)

    def _get_pack(self, name):
        return Pack.from_lazy_objects(lambda: self._load_pack_data(name), lambda: self._load_pack_index(name))

    def _upload_pack(self, basename, pack_file, index_file):
        idxblob = self.bucket.blob(posixpath.join(self.subpath, basename + '.idx'))
        datablob = self.bucket.blob(posixpath.join(self.subpath, basename + '.pack'))
        idxblob.upload_from_file(index_file)
        datablob.upload_from_file(pack_file)
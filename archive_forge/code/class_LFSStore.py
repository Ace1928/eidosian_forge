import hashlib
import os
import tempfile
class LFSStore:
    """Stores objects on disk, indexed by SHA256."""

    def __init__(self, path) -> None:
        self.path = path

    @classmethod
    def create(cls, lfs_dir):
        if not os.path.isdir(lfs_dir):
            os.mkdir(lfs_dir)
        os.mkdir(os.path.join(lfs_dir, 'tmp'))
        os.mkdir(os.path.join(lfs_dir, 'objects'))
        return cls(lfs_dir)

    @classmethod
    def from_repo(cls, repo, create=False):
        lfs_dir = os.path.join(repo.controldir, 'lfs')
        if create:
            return cls.create(lfs_dir)
        return cls(lfs_dir)

    def _sha_path(self, sha):
        return os.path.join(self.path, 'objects', sha[0:2], sha[2:4], sha)

    def open_object(self, sha):
        """Open an object by sha."""
        try:
            return open(self._sha_path(sha), 'rb')
        except FileNotFoundError as exc:
            raise KeyError(sha) from exc

    def write_object(self, chunks):
        """Write an object.

        Returns: object SHA
        """
        sha = hashlib.sha256()
        tmpdir = os.path.join(self.path, 'tmp')
        with tempfile.NamedTemporaryFile(dir=tmpdir, mode='wb', delete=False) as f:
            for chunk in chunks:
                sha.update(chunk)
                f.write(chunk)
            f.flush()
            tmppath = f.name
        path = self._sha_path(sha.hexdigest())
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        os.rename(tmppath, path)
        return sha.hexdigest()
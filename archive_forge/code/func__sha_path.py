import hashlib
import os
import tempfile
def _sha_path(self, sha):
    return os.path.join(self.path, 'objects', sha[0:2], sha[2:4], sha)
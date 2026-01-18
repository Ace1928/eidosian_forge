from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
from containerregistry.client.v2 import util
def SignedManifestToSHA256(manifest):
    """Return 'sha256:' + hex(sha256(manifest - signatures))."""
    unsigned_manifest, unused_signatures = util.DetachSignatures(manifest)
    return SHA256(unsigned_manifest.encode('utf8'))
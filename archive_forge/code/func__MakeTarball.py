from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import os.path
import tarfile
import zipfile
from googlecloudsdk.api_lib.cloudbuild import metric_names
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import files
def _MakeTarball(self, archive_path):
    """Constructs a tarball of snapshot contents.

    Args:
      archive_path: Path to place tar file.

    Returns:
      tarfile.TarFile, The constructed tar file.
    """
    tf = tarfile.open(archive_path, mode='w:gz')
    for dpath in self.dirs:
        t = tf.gettarinfo(dpath)
        if os.path.islink(dpath):
            t.type = tarfile.SYMTYPE
            t.linkname = os.readlink(dpath)
        elif os.path.isdir(dpath):
            t.type = tarfile.DIRTYPE
        else:
            log.debug('Adding [%s] as dir; os.path says is neither a dir nor a link.', dpath)
            t.type = tarfile.DIRTYPE
        t.mode = os.stat(dpath).st_mode
        tf.addfile(_ResetOwnership(t))
        log.debug('Added dir [%s]', dpath)
    for path in self.files:
        tf.add(path, filter=_ResetOwnership)
        log.debug('Added [%s]', path)
    return tf
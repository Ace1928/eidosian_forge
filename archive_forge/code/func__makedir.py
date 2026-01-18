import errno
import glob
import os
from pathlib import Path
from traitlets import Unicode, observe
from nbconvert.utils.io import link_or_copy
from .base import WriterBase
def _makedir(self, path, mode=493):
    """ensure that a directory exists

        If it doesn't exist, try to create it and protect against a race condition
        if another process is doing the same.

        The default permissions are 755, which differ from os.makedirs default of 777.
        """
    if not os.path.exists(path):
        self.log.info('Making directory %s', path)
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise OSError('%r exists but is not a directory' % path)
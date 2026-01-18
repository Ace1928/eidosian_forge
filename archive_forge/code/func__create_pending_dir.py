import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def _create_pending_dir(self):
    tmpname = '{}/{}.tmp'.format(self.path, rand_chars(10))
    try:
        self.transport.mkdir(tmpname)
    except NoSuchFile:
        self._trace('lock directory does not exist, creating it')
        self.create(mode=self._dir_modebits)
        self.transport.mkdir(tmpname)
    info = LockHeldInfo.for_this_process(self.extra_holder_info)
    self.nonce = info.nonce
    self.transport.put_bytes_non_atomic(tmpname + self.__INFO_NAME, info.to_bytes())
    return tmpname
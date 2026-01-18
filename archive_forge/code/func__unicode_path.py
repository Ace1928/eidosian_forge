import configparser
import copy
import os
import shlex
import sys
import botocore.exceptions
def _unicode_path(path):
    if isinstance(path, str):
        return path
    filesystem_encoding = sys.getfilesystemencoding()
    if filesystem_encoding is None:
        filesystem_encoding = sys.getdefaultencoding()
    return path.decode(filesystem_encoding, 'replace')
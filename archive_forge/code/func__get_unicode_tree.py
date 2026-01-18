import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def _get_unicode_tree(self):
    name0u = '0file-¶'
    name1u = '1dir-جو'
    name2u = '2file-س'
    tree = [name0u, name1u + '/', name1u + '/' + name0u, name1u + '/' + name1u + '/', name2u]
    name0 = name0u.encode('UTF-8')
    name1 = name1u.encode('UTF-8')
    name2 = name2u.encode('UTF-8')
    expected_dirblocks = [((b'', '.'), [(name0, name0, 'file', './' + name0u), (name1, name1, 'directory', './' + name1u), (name2, name2, 'file', './' + name2u)]), ((name1, './' + name1u), [(name1 + b'/' + name0, name0, 'file', './' + name1u + '/' + name0u), (name1 + b'/' + name1, name1, 'directory', './' + name1u + '/' + name1u)]), ((name1 + b'/' + name1, './' + name1u + '/' + name1u), [])]
    return (tree, expected_dirblocks)
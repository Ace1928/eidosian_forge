from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def _split_multiext(name, min=3, max=4, count=2):
    """Split a multi-part extension from a file name.

    Returns '([name minus extension], extension)'.

    Define the valid extension length (including the '.') with 'min' and 'max',
    'count' sets the number of extensions, counting from the end, to evaluate.
    Evaluation stops on the first file extension that is outside the min and max range.

    If no valid extensions are found, the original ``name`` is returned
    and ``extension`` is empty.

    :arg name: File name or path.
    :kwarg min: Minimum length of a valid file extension.
    :kwarg max: Maximum length of a valid file extension.
    :kwarg count: Number of suffixes from the end to evaluate.

    """
    extension = ''
    for i, sfx in enumerate(reversed(_suffixes(name))):
        if i >= count:
            break
        if min <= len(sfx) <= max:
            extension = '%s%s' % (sfx, extension)
            name = name.rstrip(sfx)
        else:
            break
    return (name, extension)
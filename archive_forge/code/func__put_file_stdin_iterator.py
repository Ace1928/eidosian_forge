from __future__ import (annotations, absolute_import, division, print_function)
import base64
import logging
import os
import re
import traceback
import json
import tempfile
import shlex
import subprocess
import time
import typing as t
import xml.etree.ElementTree as ET
from inspect import getfullargspec
from urllib.parse import urlunsplit
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.plugins.shell.powershell import ShellBase as PowerShellBase
from ansible.utils.hashing import secure_hash
from ansible.utils.display import Display
def _put_file_stdin_iterator(self, in_path: str, out_path: str, buffer_size: int=250000) -> t.Iterable[tuple[bytes, bool]]:
    in_size = os.path.getsize(to_bytes(in_path, errors='surrogate_or_strict'))
    offset = 0
    with open(to_bytes(in_path, errors='surrogate_or_strict'), 'rb') as in_file:
        for out_data in iter(lambda: in_file.read(buffer_size), b''):
            offset += len(out_data)
            self._display.vvvvv('WINRM PUT "%s" to "%s" (offset=%d size=%d)' % (in_path, out_path, offset, len(out_data)), host=self._winrm_host)
            b64_data = base64.b64encode(out_data) + b'\r\n'
            yield (b64_data, in_file.tell() == in_size)
        if offset == 0:
            yield (b'', True)
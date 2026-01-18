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
def _winrm_send_input(self, protocol: winrm.Protocol, shell_id: str, command_id: str, stdin: bytes, eof: bool=False) -> None:
    rq = {'env:Envelope': protocol._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Send', shell_id=shell_id)}
    stream = rq['env:Envelope'].setdefault('env:Body', {}).setdefault('rsp:Send', {}).setdefault('rsp:Stream', {})
    stream['@Name'] = 'stdin'
    stream['@CommandId'] = command_id
    stream['#text'] = base64.b64encode(to_bytes(stdin))
    if eof:
        stream['@End'] = 'true'
    protocol.send_message(xmltodict.unparse(rq))
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
def _winrm_get_raw_command_output(self, protocol: winrm.Protocol, shell_id: str, command_id: str) -> tuple[bytes, bytes, int, bool]:
    rq = {'env:Envelope': protocol._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Receive', shell_id=shell_id)}
    stream = rq['env:Envelope'].setdefault('env:Body', {}).setdefault('rsp:Receive', {}).setdefault('rsp:DesiredStream', {})
    stream['@CommandId'] = command_id
    stream['#text'] = 'stdout stderr'
    res = protocol.send_message(xmltodict.unparse(rq))
    root = ET.fromstring(res)
    stream_nodes = [node for node in root.findall('.//*') if node.tag.endswith('Stream')]
    stdout = []
    stderr = []
    return_code = -1
    for stream_node in stream_nodes:
        if not stream_node.text:
            continue
        if stream_node.attrib['Name'] == 'stdout':
            stdout.append(base64.b64decode(stream_node.text.encode('ascii')))
        elif stream_node.attrib['Name'] == 'stderr':
            stderr.append(base64.b64decode(stream_node.text.encode('ascii')))
    command_done = len([node for node in root.findall('.//*') if node.get('State', '').endswith('CommandState/Done')]) == 1
    if command_done:
        return_code = int(next((node for node in root.findall('.//*') if node.tag.endswith('ExitCode'))).text)
    return (b''.join(stdout), b''.join(stderr), return_code, command_done)
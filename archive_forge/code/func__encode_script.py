from __future__ import (absolute_import, division, print_function)
import base64
import os
import re
import shlex
import pkgutil
import xml.etree.ElementTree as ET
import ntpath
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.shell import ShellBase
def _encode_script(self, script, as_list=False, strict_mode=True, preserve_rc=True):
    """Convert a PowerShell script to a single base64-encoded command."""
    script = to_text(script)
    if script == u'-':
        cmd_parts = _common_args + ['-Command', '-']
    else:
        if strict_mode:
            script = u'Set-StrictMode -Version Latest\r\n%s' % script
        if preserve_rc:
            script = u'%s\r\nIf (-not $?) { If (Get-Variable LASTEXITCODE -ErrorAction SilentlyContinue) { exit $LASTEXITCODE } Else { exit 1 } }\r\n' % script
        script = '\n'.join([x.strip() for x in script.splitlines() if x.strip()])
        encoded_script = to_text(base64.b64encode(script.encode('utf-16-le')), 'utf-8')
        cmd_parts = _common_args + ['-EncodedCommand', encoded_script]
    if as_list:
        return cmd_parts
    return ' '.join(cmd_parts)
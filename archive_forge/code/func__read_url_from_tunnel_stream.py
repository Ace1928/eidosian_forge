import atexit
import os
import platform
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import List
import httpx
def _read_url_from_tunnel_stream(self) -> str:
    start_timestamp = time.time()
    log = []
    url = ''

    def _raise_tunnel_error():
        log_text = '\n'.join(log)
        print(log_text, file=sys.stderr)
        raise ValueError(f'{TUNNEL_ERROR_MESSAGE}\n{log_text}')
    while url == '':
        if time.time() - start_timestamp >= TUNNEL_TIMEOUT_SECONDS:
            _raise_tunnel_error()
        assert self.proc is not None
        if self.proc.stdout is None:
            continue
        line = self.proc.stdout.readline()
        line = line.decode('utf-8')
        if line == '':
            continue
        log.append(line.strip())
        if 'start proxy success' in line:
            result = re.search('start proxy success: (.+)\n', line)
            if result is None:
                _raise_tunnel_error()
            else:
                url = result.group(1)
        elif 'login to server failed' in line:
            _raise_tunnel_error()
    return url
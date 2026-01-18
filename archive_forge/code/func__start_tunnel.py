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
def _start_tunnel(self, binary: str) -> str:
    CURRENT_TUNNELS.append(self)
    command = [binary, 'http', '-n', self.share_token, '-l', str(self.local_port), '-i', self.local_host, '--uc', '--sd', 'random', '--ue', '--server_addr', f'{self.remote_host}:{self.remote_port}', '--disable_log_color']
    self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(self.kill)
    return self._read_url_from_tunnel_stream()
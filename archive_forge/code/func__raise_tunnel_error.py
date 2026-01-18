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
def _raise_tunnel_error():
    log_text = '\n'.join(log)
    print(log_text, file=sys.stderr)
    raise ValueError(f'{TUNNEL_ERROR_MESSAGE}\n{log_text}')
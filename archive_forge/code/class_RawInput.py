import argparse
import code
import gzip
import ssl
import sys
import threading
import time
import zlib
from urllib.parse import urlparse
import websocket
class RawInput:

    def raw_input(self, prompt: str='') -> str:
        line = input(prompt)
        if ENCODING and ENCODING != 'utf-8' and (not isinstance(line, str)):
            line = line.decode(ENCODING).encode('utf-8')
        elif isinstance(line, str):
            line = line.encode('utf-8')
        return line
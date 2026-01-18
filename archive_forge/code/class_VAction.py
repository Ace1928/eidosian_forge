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
class VAction(argparse.Action):

    def __call__(self, parser: argparse.Namespace, args: tuple, values: str, option_string: str=None) -> None:
        if values is None:
            values = '1'
        try:
            values = int(values)
        except ValueError:
            values = values.count('v') + 1
        setattr(args, self.dest, values)
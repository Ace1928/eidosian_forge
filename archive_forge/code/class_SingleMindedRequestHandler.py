import json
import os.path
import socket
import socketserver
import threading
from contextlib import closing, contextmanager
from http.server import SimpleHTTPRequestHandler
from typing import Callable, Generator
from urllib.request import urlopen
import h11
class SingleMindedRequestHandler(SimpleHTTPRequestHandler):

    def translate_path(self, path: str) -> str:
        return test_file_path
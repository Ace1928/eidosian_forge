import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
class HTTPGitServer(http.server.HTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, root_path) -> None:
        http.server.HTTPServer.__init__(self, server_address, GitHTTPRequestHandler)
        self.root_path = root_path
        self.server_name = 'localhost'

    def get_url(self):
        return f'http://{self.server_name}:{self.server_port}/'
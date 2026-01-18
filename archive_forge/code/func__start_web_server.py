import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def _start_web_server():
    global http_server
    from http.server import SimpleHTTPRequestHandler
    from socketserver import TCPServer
    try:
        if need_chdir:
            os.chdir(root)
            handler = SimpleHTTPRequestHandler
        else:
            handler = partial(SimpleHTTPRequestHandler, directory=root)
        http_server = TCPServer(('', 8000), handler, bind_and_activate=False)
        http_server.daemon_threads = True
        http_server.allow_reuse_address = True
        http_server.server_bind()
        http_server.server_activate()
        http_server_ready.set()
        http_server.serve_forever()
    except:
        import traceback
        traceback.print_exc()
    finally:
        http_server = None
        http_server_ready.set()
        if need_chdir:
            os.chdir(curr_dir)
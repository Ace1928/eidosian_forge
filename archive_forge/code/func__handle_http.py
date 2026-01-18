import json
import websocket
import threading
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive_web import WEB_HTML, STYLE_SHEET, FONT_AWESOME
from http.server import BaseHTTPRequestHandler, HTTPServer
def _handle_http(self, status_code, path, text=None):
    self.send_response(status_code)
    self.send_header('Content-type', 'text/html')
    self.end_headers()
    content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
    return bytes(content, 'UTF-8')
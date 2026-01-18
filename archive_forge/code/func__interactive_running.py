import json
import websocket
import threading
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive_web import WEB_HTML, STYLE_SHEET, FONT_AWESOME
from http.server import BaseHTTPRequestHandler, HTTPServer
def _interactive_running(self, reply_text):
    data = {}
    data['text'] = reply_text.decode('utf-8')
    if data['text'] == '[DONE]':
        print('[ Closing socket... ]')
        SHARED['ws'].close()
        SHARED['wb'].shutdown()
    json_data = json.dumps(data)
    SHARED['ws'].send(json_data)
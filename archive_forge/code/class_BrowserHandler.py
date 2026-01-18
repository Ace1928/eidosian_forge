import json
import websocket
import threading
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive_web import WEB_HTML, STYLE_SHEET, FONT_AWESOME
from http.server import BaseHTTPRequestHandler, HTTPServer
class BrowserHandler(BaseHTTPRequestHandler):
    """
    Handle HTTP requests.
    """

    def _interactive_running(self, reply_text):
        data = {}
        data['text'] = reply_text.decode('utf-8')
        if data['text'] == '[DONE]':
            print('[ Closing socket... ]')
            SHARED['ws'].close()
            SHARED['wb'].shutdown()
        json_data = json.dumps(data)
        SHARED['ws'].send(json_data)

    def do_HEAD(self):
        """
        Handle HEAD requests.
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST request, especially replying to a chat message.
        """
        if self.path == '/interact':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            self._interactive_running(body)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            model_response = {'id': 'Model', 'episode_done': False}
            message_available.wait()
            model_response['text'] = new_message
            message_available.clear()
            json_str = json.dumps(model_response)
            self.wfile.write(bytes(json_str, 'utf-8'))
        elif self.path == '/reset':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes('{}', 'utf-8'))
        else:
            return self._respond({'status': 500})

    def do_GET(self):
        """
        Respond to GET request, especially the initial load.
        """
        paths = {'/': {'status': 200}, '/favicon.ico': {'status': 202}}
        if self.path in paths:
            self._respond(paths[self.path])
        else:
            self._respond({'status': 500})

    def _handle_http(self, status_code, path, text=None):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
        return bytes(content, 'UTF-8')

    def _respond(self, opts):
        response = self._handle_http(opts['status'], self.path)
        self.wfile.write(response)
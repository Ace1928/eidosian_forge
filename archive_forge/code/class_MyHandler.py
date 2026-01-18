from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.image_featurizers import ImageLoader
from typing import Dict, Any
import json
import cgi
import PIL.Image as Image
from base64 import b64decode
import io
import os
class MyHandler(BaseHTTPRequestHandler):
    """
    Interactive Handler.
    """

    def interactive_running(self, data):
        """
        Generate a model response.

        :param data:
            data to send to model

        :return:
            model act dictionary
        """
        reply = {}
        reply['text'] = data['personality'][0].decode()
        img_data = str(data['image'][0])
        _, encoded = img_data.split(',', 1)
        image = Image.open(io.BytesIO(b64decode(encoded))).convert('RGB')
        reply['image'] = SHARED['image_loader'].extract(image)
        SHARED['agent'].observe(reply)
        model_res = SHARED['agent'].act()
        return model_res

    def do_HEAD(self):
        """
        Handle headers.
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST.
        """
        if self.path != '/interact':
            return self.respond({'status': 500})
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
        postvars = cgi.parse_multipart(self.rfile, pdict)
        model_response = self.interactive_running(postvars)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        json_str = json.dumps(model_response)
        self.wfile.write(bytes(json_str, 'utf-8'))

    def do_GET(self):
        """
        Handle GET.
        """
        paths = {'/': {'status': 200}, '/favicon.ico': {'status': 202}}
        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path, text=None):
        """
        Generate HTTP.
        """
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
        return bytes(content, 'UTF-8')

    def respond(self, opts):
        """
        Respond.
        """
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)
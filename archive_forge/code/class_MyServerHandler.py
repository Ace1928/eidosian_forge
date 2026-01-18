import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
from urllib.parse import unquote_plus
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
class MyServerHandler(BaseHTTPRequestHandler):

    def do_HEAD(self):
        self.send_head()

    def do_GET(self):
        global firstClient
        sp = self.path[1:]
        if unquote_plus(sp) == 'SHUTDOWN THE SERVER':
            if server_mode:
                page = 'Server must be killed with SIGTERM.'
                type = 'text/plain'
            else:
                print('Server shutting down!')
                os._exit(0)
        elif sp == '':
            type = 'text/html'
            if not server_mode and firstClient:
                firstClient = False
                page = get_static_index_page(True)
            else:
                page = get_static_index_page(False)
            word = 'green'
        elif sp.endswith('.html'):
            type = 'text/html'
            usp = unquote_plus(sp)
            if usp == 'NLTK Wordnet Browser Database Info.html':
                word = '* Database Info *'
                if os.path.isfile(usp):
                    with open(usp) as infile:
                        page = infile.read()
                else:
                    page = html_header % word + '<p>The database info file:<p><b>' + usp + '</b>' + '<p>was not found. Run this:' + '<p><b>python dbinfo_html.py</b>' + '<p>to produce it.' + html_trailer
            else:
                word = sp
                try:
                    page = get_static_page_by_path(usp)
                except FileNotFoundError:
                    page = "Internal error: Path for static page '%s' is unknown" % usp
                    type = 'text/plain'
        elif sp.startswith('search'):
            type = 'text/html'
            parts = sp.split('?')[1].split('&')
            word = [p.split('=')[1].replace('+', ' ') for p in parts if p.startswith('nextWord')][0]
            page, word = page_from_word(word)
        elif sp.startswith('lookup_'):
            type = 'text/html'
            sp = sp[len('lookup_'):]
            page, word = page_from_href(sp)
        elif sp == 'start_page':
            type = 'text/html'
            page, word = page_from_word('wordnet')
        else:
            type = 'text/plain'
            page = "Could not parse request: '%s'" % sp
        self.send_head(type)
        self.wfile.write(page.encode('utf8'))

    def send_head(self, type=None):
        self.send_response(200)
        self.send_header('Content-type', type)
        self.end_headers()

    def log_message(self, format, *args):
        global logfile
        if logfile:
            logfile.write('%s - - [%s] %s\n' % (self.address_string(), self.log_date_time_string(), format % args))
import os
from base64 import b64encode
from collections import deque
from http.client import HTTPConnection
from json import loads
from threading import Event, Thread
from time import sleep
from urllib.parse import urlparse, urlunparse
import requests
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.weakmethod import WeakMethod
def _parse_url(self, url):
    parse = urlparse(url)
    host = parse.hostname
    port = parse.port
    userpass = None
    if parse.username and parse.password:
        userpass = {'Authorization': 'Basic {}'.format(b64encode('{}:{}'.format(parse.username, parse.password).encode('utf-8')).decode('utf-8'))}
    return (host, port, userpass, parse)
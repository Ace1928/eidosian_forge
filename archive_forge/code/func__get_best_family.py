import copy
import datetime
import email.utils
import html
import http.client
import io
import itertools
import mimetypes
import os
import posixpath
import select
import shutil
import socket # For gethostbyaddr()
import socketserver
import sys
import time
import urllib.parse
from http import HTTPStatus
def _get_best_family(*address):
    infos = socket.getaddrinfo(*address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    family, type, proto, canonname, sockaddr = next(iter(infos))
    return (family, sockaddr)
import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def _auto_ssl_context():
    import OpenSSL, random
    pkey = OpenSSL.crypto.PKey()
    pkey.generate_key(OpenSSL.crypto.TYPE_RSA, 768)
    cert = OpenSSL.crypto.X509()
    cert.set_serial_number(random.randint(0, sys.maxint))
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(60 * 60 * 24 * 365)
    cert.get_subject().CN = '*'
    cert.get_subject().O = 'Dummy Certificate'
    cert.get_issuer().CN = 'Untrusted Authority'
    cert.get_issuer().O = 'Self-Signed'
    cert.set_pubkey(pkey)
    cert.sign(pkey, 'md5')
    ctx = SSL.Context(SSL.SSLv23_METHOD)
    ctx.use_privatekey(pkey)
    ctx.use_certificate(cert)
    return ctx
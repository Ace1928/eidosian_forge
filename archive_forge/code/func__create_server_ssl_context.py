import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
def _create_server_ssl_context(self, certfile, keyfile=None):
    if hasattr(ssl, 'PROTOCOL_TLS'):
        sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS)
    else:
        sslcontext = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    sslcontext.options |= ssl.OP_NO_SSLv2
    sslcontext.load_cert_chain(certfile, keyfile)
    return sslcontext
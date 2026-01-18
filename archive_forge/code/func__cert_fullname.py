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
def _cert_fullname(test_file_name, cert_file_name):
    fullname = os.path.abspath(os.path.join(os.path.dirname(test_file_name), 'certs', cert_file_name))
    assert os.path.isfile(fullname)
    return fullname
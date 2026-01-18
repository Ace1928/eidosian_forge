import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
class HTTPSOnlyHandler(HTTPSHandler, HTTPHandler):

    def http_open(self, req):
        raise URLError('Unexpected HTTP request on what should be a secure connection: %s' % req)
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
def _get_external_data(url):
    result = {}
    try:
        resp = urlopen(url)
        headers = resp.info()
        ct = headers.get('Content-Type')
        if not ct.startswith('application/json'):
            logger.debug('Unexpected response for JSON request: %s', ct)
        else:
            reader = codecs.getreader('utf-8')(resp)
            result = json.load(reader)
    except Exception as e:
        logger.exception('Failed to get external data for %s: %s', url, e)
    return result
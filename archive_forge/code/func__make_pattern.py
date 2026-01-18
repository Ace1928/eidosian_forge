import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
def _make_pattern(pat):
    if pat is None:
        return None
    if isinstance(pat, (bytes, str)):
        pat = re.compile(pat)
    if hasattr(pat, 'search'):
        return pat.search
    if callable(pat):
        return pat
    assert 0, 'Cannot make callable pattern object out of %r' % pat
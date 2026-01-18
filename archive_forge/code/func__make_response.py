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
def _make_response(self, resp, total_time):
    status, headers, body, errors = resp
    return TestResponse(self, status, headers, body, errors, total_time)
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
def _get_file_info(self, file_info):
    if len(file_info) == 2:
        filename = file_info[1]
        if self.relative_to:
            filename = os.path.join(self.relative_to, filename)
        f = open(filename, 'rb')
        content = f.read()
        f.close()
        return (file_info[0], filename, content)
    elif len(file_info) == 3:
        return file_info
    else:
        raise ValueError('upload_files need to be a list of tuples of (fieldname, filename, filecontent) or (fieldname, filename); you gave: %r' % repr(file_info)[:100])
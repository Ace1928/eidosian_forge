import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _make_response_tuple(self):
    if self.executable == 'T':
        is_exec = True
    else:
        is_exec = False
    return (self.href, self.is_dir, self.length, is_exec)
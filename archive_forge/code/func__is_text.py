import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def _is_text(data):
    return not any((b <= '\x08' or '\x0e' <= b < ' ' or b == '\x7f' for b in data))
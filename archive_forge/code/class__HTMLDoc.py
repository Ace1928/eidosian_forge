import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
class _HTMLDoc(HTMLDoc):

    def page(self, title, contents):
        """Format an HTML page."""
        css_path = 'pydoc_data/_pydoc.css'
        css_link = '<link rel="stylesheet" type="text/css" href="%s">' % css_path
        return '<!DOCTYPE>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>Pydoc: %s</title>\n%s</head><body>%s<div style="clear:both;padding-top:.5em;">%s</div>\n</body></html>' % (title, css_link, html_navbar(), contents)
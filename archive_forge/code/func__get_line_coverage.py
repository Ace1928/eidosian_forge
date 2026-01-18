from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils
def _get_line_coverage(self, coverage_xml, source_filename):
    coverage_data = None
    for entry in coverage_xml.iterfind('.//class'):
        if not entry.get('filename'):
            continue
        if entry.get('filename') == source_filename or os.path.abspath(entry.get('filename')) == source_filename:
            coverage_data = entry
            break
        elif source_filename.endswith(entry.get('filename')):
            coverage_data = entry
    if coverage_data is None:
        return None
    return dict(((int(line.get('number')), int(line.get('hits'))) for line in coverage_data.iterfind('lines/line')))
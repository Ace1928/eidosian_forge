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
def _save_annotation(self, code, generated_code, c_file=None, source_filename=None, coverage_xml=None):
    """
        lines : original cython source code split by lines
        generated_code : generated c code keyed by line number in original file
        target filename : name of the file in which to store the generated html
        c_file : filename in which the c_code has been written
        """
    if coverage_xml is not None and source_filename:
        coverage_timestamp = coverage_xml.get('timestamp', '').strip()
        covered_lines = self._get_line_coverage(coverage_xml, source_filename)
    else:
        coverage_timestamp = covered_lines = None
    annotation_items = dict(self.annotations[source_filename])
    scopes = dict(self.scopes[source_filename])
    outlist = []
    outlist.extend(self._save_annotation_header(c_file, source_filename, coverage_timestamp))
    outlist.extend(self._save_annotation_body(code, generated_code, annotation_items, scopes, covered_lines))
    outlist.extend(self._save_annotation_footer())
    return ''.join(outlist)
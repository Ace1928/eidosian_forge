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
class AnnotationItem(object):

    def __init__(self, style, text, tag='', size=0):
        self.style = style
        self.text = text
        self.tag = tag
        self.size = size

    def start(self):
        return u"<span class='cython tag %s' title='%s'>%s" % (self.style, self.text, self.tag)

    def end(self):
        return (self.size, u'</span>')
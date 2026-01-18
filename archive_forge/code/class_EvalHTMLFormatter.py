import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
class EvalHTMLFormatter(formatter.HTMLFormatter):

    def __init__(self, base_path, counter, **kw):
        super(EvalHTMLFormatter, self).__init__(**kw)
        self.base_path = base_path
        self.counter = counter

    def format_source_line(self, filename, frame):
        line = formatter.HTMLFormatter.format_source_line(self, filename, frame)
        return line + '  <a href="#" class="switch_source" tbid="%s" onClick="return showFrame(this)">&nbsp; &nbsp; <img src="%s/_debug/media/plus.jpg" border=0 width=9 height=9> &nbsp; &nbsp;</a>' % (frame.tbid, self.base_path)
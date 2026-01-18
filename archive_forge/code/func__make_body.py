import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
def _make_body(self, environ, escape):
    escape = lazify(escape)
    args = {'explanation': escape(self.explanation), 'detail': escape(self.detail or ''), 'comment': escape(self.comment or '')}
    if self.comment:
        args['html_comment'] = '<!-- %s -->' % escape(self.comment)
    else:
        args['html_comment'] = ''
    if WSGIHTTPException.body_template_obj is not self.body_template_obj:
        for k, v in environ.items():
            args[k] = escape(v)
        for k, v in self.headers.items():
            args[k.lower()] = escape(v)
    t_obj = self.body_template_obj
    return t_obj.safe_substitute(args)
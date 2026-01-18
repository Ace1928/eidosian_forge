from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
class ServerHTMLDoc(pydoc.HTMLDoc):
    """Class used to generate pydoc HTML document for a server"""

    def markup(self, text, escape=None, funcs={}, classes={}, methods={}):
        """Mark up some plain text, given a context of symbols to look for.
        Each context dictionary maps object names to anchor names."""
        escape = escape or self.escape
        results = []
        here = 0
        pattern = re.compile('\\b((http|https|ftp)://\\S+[\\w/]|RFC[- ]?(\\d+)|PEP[- ]?(\\d+)|(self\\.)?((?:\\w|\\.)+))\\b')
        while 1:
            match = pattern.search(text, here)
            if not match:
                break
            start, end = match.span()
            results.append(escape(text[here:start]))
            all, scheme, rfc, pep, selfdot, name = match.groups()
            if scheme:
                url = escape(all).replace('"', '&quot;')
                results.append('<a href="%s">%s</a>' % (url, url))
            elif rfc:
                url = 'https://www.rfc-editor.org/rfc/rfc%d.txt' % int(rfc)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif pep:
                url = 'https://peps.python.org/pep-%04d/' % int(pep)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif text[end:end + 1] == '(':
                results.append(self.namelink(name, methods, funcs, classes))
            elif selfdot:
                results.append('self.<strong>%s</strong>' % name)
            else:
                results.append(self.namelink(name, classes))
            here = end
        results.append(escape(text[here:]))
        return ''.join(results)

    def docroutine(self, object, name, mod=None, funcs={}, classes={}, methods={}, cl=None):
        """Produce HTML documentation for a function or method object."""
        anchor = (cl and cl.__name__ or '') + '-' + name
        note = ''
        title = '<a name="%s"><strong>%s</strong></a>' % (self.escape(anchor), self.escape(name))
        if callable(object):
            argspec = str(signature(object))
        else:
            argspec = '(...)'
        if isinstance(object, tuple):
            argspec = object[0] or argspec
            docstring = object[1] or ''
        else:
            docstring = pydoc.getdoc(object)
        decl = title + argspec + (note and self.grey('<font face="helvetica, arial">%s</font>' % note))
        doc = self.markup(docstring, self.preformat, funcs, classes, methods)
        doc = doc and '<dd><tt>%s</tt></dd>' % doc
        return '<dl><dt>%s</dt>%s</dl>\n' % (decl, doc)

    def docserver(self, server_name, package_documentation, methods):
        """Produce HTML documentation for an XML-RPC server."""
        fdict = {}
        for key, value in methods.items():
            fdict[key] = '#-' + key
            fdict[value] = fdict[key]
        server_name = self.escape(server_name)
        head = '<big><big><strong>%s</strong></big></big>' % server_name
        result = self.heading(head)
        doc = self.markup(package_documentation, self.preformat, fdict)
        doc = doc and '<tt>%s</tt>' % doc
        result = result + '<p>%s</p>\n' % doc
        contents = []
        method_items = sorted(methods.items())
        for key, value in method_items:
            contents.append(self.docroutine(value, key, funcs=fdict))
        result = result + self.bigsection('Methods', 'functions', ''.join(contents))
        return result

    def page(self, title, contents):
        """Format an HTML page."""
        css_path = '/pydoc.css'
        css_link = '<link rel="stylesheet" type="text/css" href="%s">' % css_path
        return '<!DOCTYPE>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>Python: %s</title>\n%s</head><body>%s</body></html>' % (title, css_link, contents)
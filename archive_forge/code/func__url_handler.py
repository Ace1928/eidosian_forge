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
def _url_handler(url, content_type='text/html'):
    """The pydoc url handler for use with the pydoc server.

    If the content_type is 'text/css', the _pydoc.css style
    sheet is read and returned if it exits.

    If the content_type is 'text/html', then the result of
    get_html_page(url) is returned.
    """

    class _HTMLDoc(HTMLDoc):

        def page(self, title, contents):
            """Format an HTML page."""
            css_path = 'pydoc_data/_pydoc.css'
            css_link = '<link rel="stylesheet" type="text/css" href="%s">' % css_path
            return '<!DOCTYPE>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>Pydoc: %s</title>\n%s</head><body>%s<div style="clear:both;padding-top:.5em;">%s</div>\n</body></html>' % (title, css_link, html_navbar(), contents)
    html = _HTMLDoc()

    def html_navbar():
        version = html.escape('%s [%s, %s]' % (platform.python_version(), platform.python_build()[0], platform.python_compiler()))
        return '\n            <div style=\'float:left\'>\n                Python %s<br>%s\n            </div>\n            <div style=\'float:right\'>\n                <div style=\'text-align:center\'>\n                  <a href="index.html">Module Index</a>\n                  : <a href="topics.html">Topics</a>\n                  : <a href="keywords.html">Keywords</a>\n                </div>\n                <div>\n                    <form action="get" style=\'display:inline;\'>\n                      <input type=text name=key size=15>\n                      <input type=submit value="Get">\n                    </form>&nbsp;\n                    <form action="search" style=\'display:inline;\'>\n                      <input type=text name=key size=15>\n                      <input type=submit value="Search">\n                    </form>\n                </div>\n            </div>\n            ' % (version, html.escape(platform.platform(terse=True)))

    def html_index():
        """Module Index page."""

        def bltinlink(name):
            return '<a href="%s.html">%s</a>' % (name, name)
        heading = html.heading('<strong class="title">Index of Modules</strong>')
        names = [name for name in sys.builtin_module_names if name != '__main__']
        contents = html.multicolumn(names, bltinlink)
        contents = [heading, '<p>' + html.bigsection('Built-in Modules', 'index', contents)]
        seen = {}
        for dir in sys.path:
            contents.append(html.index(dir, seen))
        contents.append('<p align=right class="heading-text grey"><strong>pydoc</strong> by Ka-Ping Yee&lt;ping@lfw.org&gt;</p>')
        return ('Index of Modules', ''.join(contents))

    def html_search(key):
        """Search results page."""
        search_result = []

        def callback(path, modname, desc):
            if modname[-9:] == '.__init__':
                modname = modname[:-9] + ' (package)'
            search_result.append((modname, desc and '- ' + desc))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            def onerror(modname):
                pass
            ModuleScanner().run(callback, key, onerror=onerror)

        def bltinlink(name):
            return '<a href="%s.html">%s</a>' % (name, name)
        results = []
        heading = html.heading('<strong class="title">Search Results</strong>')
        for name, desc in search_result:
            results.append(bltinlink(name) + desc)
        contents = heading + html.bigsection('key = %s' % key, 'index', '<br>'.join(results))
        return ('Search Results', contents)

    def html_topics():
        """Index of topic texts available."""

        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)
        heading = html.heading('<strong class="title">INDEX</strong>')
        names = sorted(Helper.topics.keys())
        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection('Topics', 'index', contents)
        return ('Topics', contents)

    def html_keywords():
        """Index of keywords."""
        heading = html.heading('<strong class="title">INDEX</strong>')
        names = sorted(Helper.keywords.keys())

        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)
        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection('Keywords', 'index', contents)
        return ('Keywords', contents)

    def html_topicpage(topic):
        """Topic or keyword help page."""
        buf = io.StringIO()
        htmlhelp = Helper(buf, buf)
        contents, xrefs = htmlhelp._gettopic(topic)
        if topic in htmlhelp.keywords:
            title = 'KEYWORD'
        else:
            title = 'TOPIC'
        heading = html.heading('<strong class="title">%s</strong>' % title)
        contents = '<pre>%s</pre>' % html.markup(contents)
        contents = html.bigsection(topic, 'index', contents)
        if xrefs:
            xrefs = sorted(xrefs.split())

            def bltinlink(name):
                return '<a href="topic?key=%s">%s</a>' % (name, name)
            xrefs = html.multicolumn(xrefs, bltinlink)
            xrefs = html.section('Related help topics: ', 'index', xrefs)
        return ('%s %s' % (title, topic), ''.join((heading, contents, xrefs)))

    def html_getobj(url):
        obj = locate(url, forceload=1)
        if obj is None and url != 'None':
            raise ValueError('could not find object')
        title = describe(obj)
        content = html.document(obj, url)
        return (title, content)

    def html_error(url, exc):
        heading = html.heading('<strong class="title">Error</strong>')
        contents = '<br>'.join((html.escape(line) for line in format_exception_only(type(exc), exc)))
        contents = heading + html.bigsection(url, 'error', contents)
        return ('Error - %s' % url, contents)

    def get_html_page(url):
        """Generate an HTML page for url."""
        complete_url = url
        if url.endswith('.html'):
            url = url[:-5]
        try:
            if url in ('', 'index'):
                title, content = html_index()
            elif url == 'topics':
                title, content = html_topics()
            elif url == 'keywords':
                title, content = html_keywords()
            elif '=' in url:
                op, _, url = url.partition('=')
                if op == 'search?key':
                    title, content = html_search(url)
                elif op == 'topic?key':
                    try:
                        title, content = html_topicpage(url)
                    except ValueError:
                        title, content = html_getobj(url)
                elif op == 'get?key':
                    if url in ('', 'index'):
                        title, content = html_index()
                    else:
                        try:
                            title, content = html_getobj(url)
                        except ValueError:
                            title, content = html_topicpage(url)
                else:
                    raise ValueError('bad pydoc url')
            else:
                title, content = html_getobj(url)
        except Exception as exc:
            title, content = html_error(complete_url, exc)
        return html.page(title, content)
    if url.startswith('/'):
        url = url[1:]
    if content_type == 'text/css':
        path_here = os.path.dirname(os.path.realpath(__file__))
        css_path = os.path.join(path_here, url)
        with open(css_path) as fp:
            return ''.join(fp.readlines())
    elif content_type == 'text/html':
        return get_html_page(url)
    raise TypeError('unknown content type %r for url %s' % (content_type, url))
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
class EvalException(object):

    def __init__(self, application, global_conf=None, xmlhttp_key=None):
        self.application = application
        self.debug_infos = {}
        if xmlhttp_key is None:
            if global_conf is None:
                xmlhttp_key = '_'
            else:
                xmlhttp_key = global_conf.get('xmlhttp_key', '_')
        self.xmlhttp_key = xmlhttp_key

    def __call__(self, environ, start_response):
        assert not environ['wsgi.multiprocess'], 'The EvalException middleware is not usable in a multi-process environment'
        environ['paste.evalexception'] = self
        if environ.get('PATH_INFO', '').startswith('/_debug/'):
            return self.debug(environ, start_response)
        else:
            return self.respond(environ, start_response)

    def debug(self, environ, start_response):
        assert request.path_info_pop(environ) == '_debug'
        next_part = request.path_info_pop(environ)
        method = getattr(self, next_part, None)
        if not method:
            exc = httpexceptions.HTTPNotFound('%r not found when parsing %r' % (next_part, wsgilib.construct_url(environ)))
            return exc.wsgi_application(environ, start_response)
        if not getattr(method, 'exposed', False):
            exc = httpexceptions.HTTPForbidden('%r not allowed' % next_part)
            return exc.wsgi_application(environ, start_response)
        return method(environ, start_response)

    def media(self, environ, start_response):
        """
        Static path where images and other files live
        """
        app = urlparser.StaticURLParser(os.path.join(os.path.dirname(__file__), 'media'))
        return app(environ, start_response)
    media.exposed = True

    def mochikit(self, environ, start_response):
        """
        Static path where MochiKit lives
        """
        app = urlparser.StaticURLParser(os.path.join(os.path.dirname(__file__), 'mochikit'))
        return app(environ, start_response)
    mochikit.exposed = True

    def summary(self, environ, start_response):
        """
        Returns a JSON-format summary of all the cached
        exception reports
        """
        start_response('200 OK', [('Content-type', 'text/x-json')])
        data = []
        items = self.debug_infos.values()
        items.sort(lambda a, b: cmp(a.created, b.created))
        data = [item.json() for item in items]
        return [repr(data)]
    summary.exposed = True

    def view(self, environ, start_response):
        """
        View old exception reports
        """
        id = int(request.path_info_pop(environ))
        if id not in self.debug_infos:
            start_response('500 Server Error', [('Content-type', 'text/html')])
            return ['Traceback by id %s does not exist (maybe the server has been restarted?)' % id]
        debug_info = self.debug_infos[id]
        return debug_info.wsgi_application(environ, start_response)
    view.exposed = True

    def make_view_url(self, environ, base_path, count):
        return base_path + '/_debug/view/%s' % count

    def show_frame(self, tbid, debug_info, **kw):
        frame = debug_info.frame(int(tbid))
        vars = frame.tb_frame.f_locals
        if vars:
            registry.restorer.restoration_begin(debug_info.counter)
            local_vars = make_table(vars)
            registry.restorer.restoration_end()
        else:
            local_vars = 'No local vars'
        return input_form(tbid, debug_info) + local_vars
    show_frame = wsgiapp()(get_debug_info(show_frame))

    def exec_input(self, tbid, debug_info, input, **kw):
        if not input.strip():
            return ''
        input = input.rstrip() + '\n'
        frame = debug_info.frame(int(tbid))
        vars = frame.tb_frame.f_locals
        glob_vars = frame.tb_frame.f_globals
        context = evalcontext.EvalContext(vars, glob_vars)
        registry.restorer.restoration_begin(debug_info.counter)
        output = context.exec_expr(input)
        registry.restorer.restoration_end()
        input_html = formatter.str2html(input)
        return '<code style="color: #060">&gt;&gt;&gt;</code> <code>%s</code><br>\n%s' % (preserve_whitespace(input_html, quote=False), preserve_whitespace(output))
    exec_input = wsgiapp()(get_debug_info(exec_input))

    def respond(self, environ, start_response):
        if environ.get('paste.throw_errors'):
            return self.application(environ, start_response)
        base_path = request.construct_url(environ, with_path_info=False, with_query_string=False)
        environ['paste.throw_errors'] = True
        started = []

        def detect_start_response(status, headers, exc_info=None):
            try:
                return start_response(status, headers, exc_info)
            except:
                raise
            else:
                started.append(True)
        try:
            __traceback_supplement__ = (errormiddleware.Supplement, self, environ)
            app_iter = self.application(environ, detect_start_response)
            try:
                return_iter = list(app_iter)
                return return_iter
            finally:
                if hasattr(app_iter, 'close'):
                    app_iter.close()
        except:
            exc_info = sys.exc_info()
            for expected in environ.get('paste.expected_exceptions', []):
                if isinstance(exc_info[1], expected):
                    raise
            registry.restorer.save_registry_state(environ)
            count = get_debug_count(environ)
            view_uri = self.make_view_url(environ, base_path, count)
            if not started:
                headers = [('content-type', 'text/html')]
                headers.append(('X-Debug-URL', view_uri))
                start_response('500 Internal Server Error', headers, exc_info)
            msg = 'Debug at: %s\n' % view_uri
            environ['wsgi.errors'].write(msg)
            exc_data = collector.collect_exception(*exc_info)
            debug_info = DebugInfo(count, exc_info, exc_data, base_path, environ, view_uri)
            assert count not in self.debug_infos
            self.debug_infos[count] = debug_info
            if self.xmlhttp_key:
                get_vars = request.parse_querystring(environ)
                if dict(get_vars).get(self.xmlhttp_key):
                    exc_data = collector.collect_exception(*exc_info)
                    html = formatter.format_html(exc_data, include_hidden_frames=False, include_reusable=False, show_extra_data=False)
                    return [html]
            return debug_info.content()

    def exception_handler(self, exc_info, environ):
        simple_html_error = False
        if self.xmlhttp_key:
            get_vars = request.parse_querystring(environ)
            if dict(get_vars).get(self.xmlhttp_key):
                simple_html_error = True
        return errormiddleware.handle_exception(exc_info, environ['wsgi.errors'], html=True, debug_mode=True, simple_html_error=simple_html_error)
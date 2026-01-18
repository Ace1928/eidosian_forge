import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
class LogManager(object):
    """An object to assist both simple and advanced logging.

    ``cherrypy.log`` is an instance of this class.
    """
    appid = None
    'The id() of the Application object which owns this log manager. If this\n    is a global log manager, appid is None.'
    error_log = None
    'The actual :class:`logging.Logger` instance for error messages.'
    access_log = None
    'The actual :class:`logging.Logger` instance for access messages.'
    access_log_format = '{h} {l} {u} {t} "{r}" {s} {b} "{f}" "{a}"'
    logger_root = None
    'The "top-level" logger name.\n\n    This string will be used as the first segment in the Logger names.\n    The default is "cherrypy", for example, in which case the Logger names\n    will be of the form::\n\n        cherrypy.error.<appid>\n        cherrypy.access.<appid>\n    '

    def __init__(self, appid=None, logger_root='cherrypy'):
        self.logger_root = logger_root
        self.appid = appid
        if appid is None:
            self.error_log = logging.getLogger('%s.error' % logger_root)
            self.access_log = logging.getLogger('%s.access' % logger_root)
        else:
            self.error_log = logging.getLogger('%s.error.%s' % (logger_root, appid))
            self.access_log = logging.getLogger('%s.access.%s' % (logger_root, appid))
        self.error_log.setLevel(logging.INFO)
        self.access_log.setLevel(logging.INFO)
        self.error_log.addHandler(NullHandler())
        self.access_log.addHandler(NullHandler())
        cherrypy.engine.subscribe('graceful', self.reopen_files)

    def reopen_files(self):
        """Close and reopen all file handlers."""
        for log in (self.error_log, self.access_log):
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    h.acquire()
                    h.stream.close()
                    h.stream = open(h.baseFilename, h.mode)
                    h.release()

    def error(self, msg='', context='', severity=logging.INFO, traceback=False):
        """Write the given ``msg`` to the error log.

        This is not just for errors! Applications may call this at any time
        to log application-specific information.

        If ``traceback`` is True, the traceback of the current exception
        (if any) will be appended to ``msg``.
        """
        exc_info = None
        if traceback:
            exc_info = _cperror._exc_info()
        self.error_log.log(severity, ' '.join((self.time(), context, msg)), exc_info=exc_info)

    def __call__(self, *args, **kwargs):
        """An alias for ``error``."""
        return self.error(*args, **kwargs)

    def access(self):
        """Write to the access log (in Apache/NCSA Combined Log format).

        See the
        `apache documentation
        <http://httpd.apache.org/docs/current/logs.html#combined>`_
        for format details.

        CherryPy calls this automatically for you. Note there are no arguments;
        it collects the data itself from
        :class:`cherrypy.request<cherrypy._cprequest.Request>`.

        Like Apache started doing in 2.0.46, non-printable and other special
        characters in %r (and we expand that to all parts) are escaped using
        \\xhh sequences, where hh stands for the hexadecimal representation
        of the raw byte. Exceptions from this rule are " and \\, which are
        escaped by prepending a backslash, and all whitespace characters,
        which are written in their C-style notation (\\n, \\t, etc).
        """
        request = cherrypy.serving.request
        remote = request.remote
        response = cherrypy.serving.response
        outheaders = response.headers
        inheaders = request.headers
        if response.output_status is None:
            status = '-'
        else:
            status = response.output_status.split(b' ', 1)[0]
            status = status.decode('ISO-8859-1')
        atoms = {'h': remote.name or remote.ip, 'l': '-', 'u': getattr(request, 'login', None) or '-', 't': self.time(), 'r': request.request_line, 's': status, 'b': dict.get(outheaders, 'Content-Length', '') or '-', 'f': dict.get(inheaders, 'Referer', ''), 'a': dict.get(inheaders, 'User-Agent', ''), 'o': dict.get(inheaders, 'Host', '-'), 'i': request.unique_id, 'z': LazyRfc3339UtcTime()}
        for k, v in atoms.items():
            if not isinstance(v, str):
                v = str(v)
            v = v.replace('"', '\\"').encode('utf8')
            v = repr(v)[2:-1]
            v = v.replace('\\\\', '\\')
            atoms[k] = v
        try:
            self.access_log.log(logging.INFO, self.access_log_format.format(**atoms))
        except Exception:
            self(traceback=True)

    def time(self):
        """Return now() in Apache Common Log Format (no timezone)."""
        now = datetime.datetime.now()
        monthnames = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month = monthnames[now.month - 1].capitalize()
        return '[%02d/%s/%04d:%02d:%02d:%02d]' % (now.day, month, now.year, now.hour, now.minute, now.second)

    def _get_builtin_handler(self, log, key):
        for h in log.handlers:
            if getattr(h, '_cpbuiltin', None) == key:
                return h

    def _set_screen_handler(self, log, enable, stream=None):
        h = self._get_builtin_handler(log, 'screen')
        if enable:
            if not h:
                if stream is None:
                    stream = sys.stderr
                h = logging.StreamHandler(stream)
                h.setFormatter(logfmt)
                h._cpbuiltin = 'screen'
                log.addHandler(h)
        elif h:
            log.handlers.remove(h)

    @property
    def screen(self):
        """Turn stderr/stdout logging on or off.

        If you set this to True, it'll add the appropriate StreamHandler for
        you. If you set it to False, it will remove the handler.
        """
        h = self._get_builtin_handler
        has_h = h(self.error_log, 'screen') or h(self.access_log, 'screen')
        return bool(has_h)

    @screen.setter
    def screen(self, newvalue):
        self._set_screen_handler(self.error_log, newvalue, stream=sys.stderr)
        self._set_screen_handler(self.access_log, newvalue, stream=sys.stdout)

    def _add_builtin_file_handler(self, log, fname):
        h = logging.FileHandler(fname)
        h.setFormatter(logfmt)
        h._cpbuiltin = 'file'
        log.addHandler(h)

    def _set_file_handler(self, log, filename):
        h = self._get_builtin_handler(log, 'file')
        if filename:
            if h:
                if h.baseFilename != os.path.abspath(filename):
                    h.close()
                    log.handlers.remove(h)
                    self._add_builtin_file_handler(log, filename)
            else:
                self._add_builtin_file_handler(log, filename)
        elif h:
            h.close()
            log.handlers.remove(h)

    @property
    def error_file(self):
        """The filename for self.error_log.

        If you set this to a string, it'll add the appropriate FileHandler for
        you. If you set it to ``None`` or ``''``, it will remove the handler.
        """
        h = self._get_builtin_handler(self.error_log, 'file')
        if h:
            return h.baseFilename
        return ''

    @error_file.setter
    def error_file(self, newvalue):
        self._set_file_handler(self.error_log, newvalue)

    @property
    def access_file(self):
        """The filename for self.access_log.

        If you set this to a string, it'll add the appropriate FileHandler for
        you. If you set it to ``None`` or ``''``, it will remove the handler.
        """
        h = self._get_builtin_handler(self.access_log, 'file')
        if h:
            return h.baseFilename
        return ''

    @access_file.setter
    def access_file(self, newvalue):
        self._set_file_handler(self.access_log, newvalue)

    def _set_wsgi_handler(self, log, enable):
        h = self._get_builtin_handler(log, 'wsgi')
        if enable:
            if not h:
                h = WSGIErrorHandler()
                h.setFormatter(logfmt)
                h._cpbuiltin = 'wsgi'
                log.addHandler(h)
        elif h:
            log.handlers.remove(h)

    @property
    def wsgi(self):
        """Write errors to wsgi.errors.

        If you set this to True, it'll add the appropriate
        :class:`WSGIErrorHandler<cherrypy._cplogging.WSGIErrorHandler>` for you
        (which writes errors to ``wsgi.errors``).
        If you set it to False, it will remove the handler.
        """
        return bool(self._get_builtin_handler(self.error_log, 'wsgi'))

    @wsgi.setter
    def wsgi(self, newvalue):
        self._set_wsgi_handler(self.error_log, newvalue)
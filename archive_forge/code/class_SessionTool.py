import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
class SessionTool(Tool):
    """Session Tool for CherryPy.

    sessions.locking
        When 'implicit' (the default), the session will be locked for you,
        just before running the page handler.

        When 'early', the session will be locked before reading the request
        body. This is off by default for safety reasons; for example,
        a large upload would block the session, denying an AJAX
        progress meter
        (`issue <https://github.com/cherrypy/cherrypy/issues/630>`_).

        When 'explicit' (or any other value), you need to call
        cherrypy.session.acquire_lock() yourself before using
        session data.
    """

    def __init__(self):
        Tool.__init__(self, 'before_request_body', _sessions.init)

    def _lock_session(self):
        cherrypy.serving.session.acquire_lock()

    def _setup(self):
        """Hook this tool into cherrypy.request.

        The standard CherryPy request object will automatically call this
        method when the tool is "turned on" in config.
        """
        hooks = cherrypy.serving.request.hooks
        conf = self._merged_args()
        p = conf.pop('priority', None)
        if p is None:
            p = getattr(self.callable, 'priority', self._priority)
        hooks.attach(self._point, self.callable, priority=p, **conf)
        locking = conf.pop('locking', 'implicit')
        if locking == 'implicit':
            hooks.attach('before_handler', self._lock_session)
        elif locking == 'early':
            hooks.attach('before_request_body', self._lock_session, priority=60)
        else:
            pass
        hooks.attach('before_finalize', _sessions.save)
        hooks.attach('on_end_request', _sessions.close)

    def regenerate(self):
        """Drop the current session and make a new one (with a new id)."""
        sess = cherrypy.serving.session
        sess.regenerate()
        relevant = ('path', 'path_header', 'name', 'timeout', 'domain', 'secure')
        conf = dict(((k, v) for k, v in self._merged_args().items() if k in relevant))
        _sessions.set_response_cookie(**conf)
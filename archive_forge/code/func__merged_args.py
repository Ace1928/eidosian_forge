import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
def _merged_args(self, d=None):
    """Return a dict of configuration entries for this Tool."""
    if d:
        conf = d.copy()
    else:
        conf = {}
    tm = cherrypy.serving.request.toolmaps[self.namespace]
    if self._name in tm:
        conf.update(tm[self._name])
    if 'on' in conf:
        del conf['on']
    return conf
import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
class XMLRPCController(object):
    """A Controller (page handler collection) for XML-RPC.

    To use it, have your controllers subclass this base class (it will
    turn on the tool for you).

    You can also supply the following optional config entries::

        tools.xmlrpc.encoding: 'utf-8'
        tools.xmlrpc.allow_none: 0

    XML-RPC is a rather discontinuous layer over HTTP; dispatching to the
    appropriate handler must first be performed according to the URL, and
    then a second dispatch step must take place according to the RPC method
    specified in the request body. It also allows a superfluous "/RPC2"
    prefix in the URL, supplies its own handler args in the body, and
    requires a 200 OK "Fault" response instead of 404 when the desired
    method is not found.

    Therefore, XML-RPC cannot be implemented for CherryPy via a Tool alone.
    This Controller acts as the dispatch target for the first half (based
    on the URL); it then reads the RPC method from the request body and
    does its own second dispatch step based on that method. It also reads
    body params, and returns a Fault on error.

    The XMLRPCDispatcher strips any /RPC2 prefix; if you aren't using /RPC2
    in your URL's, you can safely skip turning on the XMLRPCDispatcher.
    Otherwise, you need to use declare it in config::

        request.dispatch: cherrypy.dispatch.XMLRPCDispatcher()
    """
    _cp_config = {'tools.xmlrpc.on': True}

    @expose
    def default(self, *vpath, **params):
        rpcparams, rpcmethod = _xmlrpc.process_body()
        subhandler = self
        for attr in str(rpcmethod).split('.'):
            subhandler = getattr(subhandler, attr, None)
        if subhandler and getattr(subhandler, 'exposed', False):
            body = subhandler(*vpath + rpcparams, **params)
        else:
            raise Exception('method "%s" is not supported' % attr)
        conf = cherrypy.serving.request.toolmaps['tools'].get('xmlrpc', {})
        _xmlrpc.respond(body, conf.get('encoding', 'utf-8'), conf.get('allow_none', 0))
        return cherrypy.serving.response.body
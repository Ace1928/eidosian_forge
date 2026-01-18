import string
import sys
import types
import cherrypy
def XMLRPCDispatcher(next_dispatcher=Dispatcher()):
    from cherrypy.lib import xmlrpcutil

    def xmlrpc_dispatch(path_info):
        path_info = xmlrpcutil.patched_path(path_info)
        return next_dispatcher(path_info)
    return xmlrpc_dispatch
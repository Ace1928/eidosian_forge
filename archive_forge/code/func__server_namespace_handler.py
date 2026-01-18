import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import reprconf
def _server_namespace_handler(k, v):
    """Config handler for the "server" namespace."""
    atoms = k.split('.', 1)
    if len(atoms) > 1:
        if not hasattr(cherrypy, 'servers'):
            cherrypy.servers = {}
        servername, k = atoms
        if servername not in cherrypy.servers:
            from cherrypy import _cpserver
            cherrypy.servers[servername] = _cpserver.Server()
            cherrypy.servers[servername].subscribe()
        if k == 'on':
            if v:
                cherrypy.servers[servername].subscribe()
            else:
                cherrypy.servers[servername].unsubscribe()
        else:
            setattr(cherrypy.servers[servername], k, v)
    else:
        setattr(cherrypy.server, k, v)
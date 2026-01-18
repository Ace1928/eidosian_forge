import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import reprconf
def _engine_namespace_handler(k, v):
    """Config handler for the "engine" namespace."""
    engine = cherrypy.engine
    if k in {'SIGHUP', 'SIGTERM'}:
        engine.subscribe(k, v)
        return
    if '.' in k:
        plugin, attrname = k.split('.', 1)
        plugin = getattr(engine, plugin)
        op = 'subscribe' if v else 'unsubscribe'
        sub_unsub = getattr(plugin, op, None)
        if attrname == 'on' and callable(sub_unsub):
            sub_unsub()
            return
        setattr(plugin, attrname, v)
    else:
        setattr(engine, k, v)
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import reprconf
def _tree_namespace_handler(k, v):
    """Namespace handler for the 'tree' config namespace."""
    if isinstance(v, dict):
        for script_name, app in v.items():
            cherrypy.tree.graft(app, script_name)
            msg = 'Mounted: %s on %s' % (app, script_name or '/')
            cherrypy.engine.log(msg)
    else:
        cherrypy.tree.graft(v, v.script_name)
        cherrypy.engine.log('Mounted: %s on %s' % (v, v.script_name or '/'))
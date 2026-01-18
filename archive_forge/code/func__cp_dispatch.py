import cherrypy
from cherrypy.test import helper
def _cp_dispatch(self, vpath):
    """Make sure that popping ALL of vpath still shows the index
            handler.
            """
    while vpath:
        vpath.pop()
    return self
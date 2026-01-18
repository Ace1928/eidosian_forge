import os.path
import cherrypy
class HitCounter:
    _cp_config = {'tools.sessions.on': True}

    @cherrypy.expose
    def index(self):
        count = cherrypy.session.get('count', 0) + 1
        cherrypy.session['count'] = count
        return "\n            During your current session, you've viewed this\n            page %s times! Your life is a patio of fun!\n        " % count
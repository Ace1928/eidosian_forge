import os.path
import cherrypy
class UsersPage:

    @cherrypy.expose
    def index(self):
        return '\n            <a href="./remi">Remi Delon</a><br/>\n            <a href="./hendrik">Hendrik Mans</a><br/>\n            <a href="./lorenzo">Lorenzo Lamas</a><br/>\n        '

    @cherrypy.expose
    def default(self, user):
        if user == 'remi':
            out = 'Remi Delon, CherryPy lead developer'
        elif user == 'hendrik':
            out = 'Hendrik Mans, CherryPy co-developer & crazy German'
        elif user == 'lorenzo':
            out = 'Lorenzo Lamas, famous actor and singer!'
        else:
            out = 'Unknown user. :-('
        return '%s (<a href="./">back</a>)' % out
import os.path
import cherrypy
class AnotherPage(Page):
    title = 'Another Page'

    @cherrypy.expose
    def index(self):
        return self.header() + '\n            <p>\n            And this is the amazing second page!\n            </p>\n        ' + self.footer()
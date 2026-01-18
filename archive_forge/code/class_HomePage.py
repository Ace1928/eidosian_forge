import os.path
import cherrypy
class HomePage(Page):
    title = 'Tutorial 5'

    def __init__(self):
        self.another = AnotherPage()

    @cherrypy.expose
    def index(self):
        return self.header() + '\n            <p>\n            Isn\'t this exciting? There\'s\n            <a href="./another/">another page</a>, too!\n            </p>\n        ' + self.footer()
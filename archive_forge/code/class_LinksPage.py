import os.path
import cherrypy
class LinksPage:

    def __init__(self):
        self.extra = ExtraLinksPage()

    @cherrypy.expose
    def index(self):
        return '\n            <p>Here are some useful links:</p>\n\n            <ul>\n                <li>\n                    <a href="http://www.cherrypy.dev">The CherryPy Homepage</a>\n                </li>\n                <li>\n                    <a href="http://www.python.org">The Python Homepage</a>\n                </li>\n            </ul>\n\n            <p>You can check out some extra useful\n            links <a href="./extra/">here</a>.</p>\n\n            <p>[<a href="../">Return</a>]</p>\n        '
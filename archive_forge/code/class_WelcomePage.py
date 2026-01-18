import os.path
import cherrypy
class WelcomePage:

    @cherrypy.expose
    def index(self):
        return '\n            <form action="greetUser" method="GET">\n            What is your name?\n            <input type="text" name="name" />\n            <input type="submit" />\n            </form>'

    @cherrypy.expose
    def greetUser(self, name=None):
        if name:
            return "Hey %s, what's up?" % name
        elif name is None:
            return 'Please enter your name <a href="./">here</a>.'
        else:
            return 'No, really, enter your name <a href="./">here</a>.'
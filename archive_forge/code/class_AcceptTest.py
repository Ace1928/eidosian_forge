import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
class AcceptTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_Accept_Tool(self):
        self.getPage('/accept/feed')
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', 'application/atom+xml')])
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', 'application/*')])
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', '*/*')])
        self.assertStatus(200)
        self.assertInBody('<title>Unknown Blog</title>')
        self.getPage('/accept/feed', headers=[('Accept', 'text/html')])
        self.assertErrorPage(406, 'Your client sent this Accept header: text/html. But this resource only emits these media types: application/atom+xml.')
        self.getPage('/accept/')
        self.assertStatus(200)
        self.assertBody('<a href="feed">Atom feed</a>')

    def test_accept_selection(self):
        self.getPage('/accept/select', [('Accept', 'text/html')])
        self.assertStatus(200)
        self.assertBody('<h2>Page Title</h2>')
        self.getPage('/accept/select', [('Accept', 'text/plain')])
        self.assertStatus(200)
        self.assertBody('PAGE TITLE')
        self.getPage('/accept/select', [('Accept', 'text/plain, text/*;q=0.5')])
        self.assertStatus(200)
        self.assertBody('PAGE TITLE')
        self.getPage('/accept/select', [('Accept', 'text/*')])
        self.assertStatus(200)
        self.assertBody('<h2>Page Title</h2>')
        self.getPage('/accept/select', [('Accept', '*/*')])
        self.assertStatus(200)
        self.assertBody('<h2>Page Title</h2>')
        self.getPage('/accept/select', [('Accept', 'application/xml')])
        self.assertErrorPage(406, 'Your client sent this Accept header: application/xml. But this resource only emits these media types: text/html, text/plain.')
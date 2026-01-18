import os
import platform
import threading
import time
from http.client import HTTPConnection
from distutils.spawn import find_executable
import pytest
from path import Path
from more_itertools import consume
import portend
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.lib import sessions
from cherrypy.lib import reprconf
from cherrypy.lib.httputil import response_codes
from cherrypy.test import helper
from cherrypy import _json as json
@pytest.mark.skipif(platform.system() == 'Windows', reason='pytest-services helper does not work under Windows')
@pytest.mark.usefixtures('memcached_configured')
class MemcachedSessionTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_0_Session(self):
        self.getPage('/set_session_cls/cherrypy.lib.sessions.MemcachedSession')
        self.getPage('/testStr')
        assert self.body == b'1'
        self.getPage('/testGen', self.cookies)
        assert self.body == b'2'
        self.getPage('/testStr', self.cookies)
        assert self.body == b'3'
        self.getPage('/length', self.cookies)
        self.assertErrorPage(500)
        assert b'NotImplementedError' in self.body
        self.getPage('/delkey?key=counter', self.cookies)
        assert self.status_code == 200
        time.sleep(1.25)
        self.getPage('/')
        assert self.body == b'1'
        self.getPage('/keyin?key=counter', self.cookies)
        assert self.body == b'True'
        self.getPage('/delete', self.cookies)
        assert self.body == b'done'

    def test_1_Concurrency(self):
        client_thread_count = 5
        request_count = 30
        self.getPage('/')
        assert self.body == b'1'
        cookies = self.cookies
        data_dict = {}

        def request(index):
            for i in range(request_count):
                self.getPage('/', cookies)
            if not self.body.isdigit():
                self.fail(self.body)
            data_dict[index] = int(self.body)
        ts = []
        for c in range(client_thread_count):
            data_dict[c] = 0
            t = threading.Thread(target=request, args=(c,))
            ts.append(t)
            t.start()
        for t in ts:
            t.join()
        hitcount = max(data_dict.values())
        expected = 1 + client_thread_count * request_count
        assert hitcount == expected

    def test_3_Redirect(self):
        self.getPage('/testStr')
        self.getPage('/iredir', self.cookies)
        assert self.body == b'MemcachedSession'

    def test_5_Error_paths(self):
        self.getPage('/unknown/page')
        self.assertErrorPage(404, "The path '/unknown/page' was not found.")
        self.getPage('/restricted', self.cookies, method='POST')
        self.assertErrorPage(405, response_codes[405][1])
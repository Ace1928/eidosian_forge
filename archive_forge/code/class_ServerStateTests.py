import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
class ServerStateTests(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def setUp(self):
        cherrypy.server.socket_timeout = 0.1
        self.do_gc_test = False

    def test_0_NormalStateFlow(self):
        engine.stop()
        self.assertEqual(db_connection.running, False)
        self.assertEqual(db_connection.startcount, 1)
        self.assertEqual(len(db_connection.threads), 0)
        engine.start()
        self.assertEqual(engine.state, engine.states.STARTED)
        host = cherrypy.server.socket_host
        port = cherrypy.server.socket_port
        portend.occupied(host, port, timeout=0.1)
        self.assertEqual(db_connection.running, True)
        self.assertEqual(db_connection.startcount, 2)
        self.assertEqual(len(db_connection.threads), 0)
        self.getPage('/')
        self.assertBody('Hello World')
        self.assertEqual(len(db_connection.threads), 1)
        engine.stop()
        self.assertEqual(engine.state, engine.states.STOPPED)
        self.assertEqual(db_connection.running, False)
        self.assertEqual(len(db_connection.threads), 0)

        def exittest():
            self.getPage('/')
            self.assertBody('Hello World')
            engine.exit()
        cherrypy.server.start()
        engine.start_with_callback(exittest)
        engine.block()
        self.assertEqual(engine.state, engine.states.EXITING)

    def test_1_Restart(self):
        cherrypy.server.start()
        engine.start()
        self.assertEqual(db_connection.running, True)
        grace = db_connection.gracecount
        self.getPage('/')
        self.assertBody('Hello World')
        self.assertEqual(len(db_connection.threads), 1)
        engine.graceful()
        self.assertEqual(engine.state, engine.states.STARTED)
        self.getPage('/')
        self.assertBody('Hello World')
        self.assertEqual(db_connection.running, True)
        self.assertEqual(db_connection.gracecount, grace + 1)
        self.assertEqual(len(db_connection.threads), 1)
        self.getPage('/graceful')
        self.assertEqual(engine.state, engine.states.STARTED)
        self.assertBody('app was (gracefully) restarted succesfully')
        self.assertEqual(db_connection.running, True)
        self.assertEqual(db_connection.gracecount, grace + 2)
        self.assertEqual(len(db_connection.threads), 0)
        engine.stop()
        self.assertEqual(engine.state, engine.states.STOPPED)
        self.assertEqual(db_connection.running, False)
        self.assertEqual(len(db_connection.threads), 0)

    def test_2_KeyboardInterrupt(self):
        engine.start()
        cherrypy.server.start()
        self.persistent = True
        try:
            self.getPage('/')
            self.assertStatus('200 OK')
            self.assertBody('Hello World')
            self.assertNoHeader('Connection')
            cherrypy.server.httpserver.interrupt = KeyboardInterrupt
            engine.block()
            self.assertEqual(db_connection.running, False)
            self.assertEqual(len(db_connection.threads), 0)
            self.assertEqual(engine.state, engine.states.EXITING)
        finally:
            self.persistent = False
        engine.start()
        cherrypy.server.start()
        try:
            self.getPage('/ctrlc', raise_subcls=BadStatusLine)
        except BadStatusLine:
            pass
        else:
            print(self.body)
            self.fail('AssertionError: BadStatusLine not raised')
        engine.block()
        self.assertEqual(db_connection.running, False)
        self.assertEqual(len(db_connection.threads), 0)

    @pytest.mark.xfail('sys.platform == "Darwin" and sys.version_info > (3, 7) and os.environ["TRAVIS"]', reason='https://github.com/cherrypy/cherrypy/issues/1693')
    def test_4_Autoreload(self):
        if engine.state != engine.states.EXITING:
            engine.exit()
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
        p.write_conf(extra='test_case_name: "test_4_Autoreload"')
        p.start(imports='cherrypy.test._test_states_demo')
        try:
            self.getPage('/start')
            start = float(self.body)
            time.sleep(2)
            os.utime(os.path.join(thisdir, '_test_states_demo.py'), None)
            time.sleep(2)
            host = cherrypy.server.socket_host
            port = cherrypy.server.socket_port
            portend.occupied(host, port, timeout=5)
            self.getPage('/start')
            if not float(self.body) > start:
                raise AssertionError('start time %s not greater than %s' % (float(self.body), start))
        finally:
            self.getPage('/exit')
        p.join()

    def test_5_Start_Error(self):
        if engine.state != engine.states.EXITING:
            engine.exit()
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https', wait=True)
        p.write_conf(extra='starterror: True\ntest_case_name: "test_5_Start_Error"\n')
        p.start(imports='cherrypy.test._test_states_demo')
        if p.exit_code == 0:
            self.fail('Process failed to return nonzero exit code.')
import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
class TestHTTPSilentRedirections(http_utils.TestCaseWithRedirectedWebserver):
    """Test redirections.

    http implementations do not redirect silently anymore (they
    do not redirect at all in fact). The mechanism is still in
    place at the Request level and these tests
    exercise it.
    """
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())

    def setUp(self):
        super().setUp()
        install_redirected_request(self)
        cleanup_http_redirection_connections(self)
        self.build_tree_contents([('a', b'a'), ('1/',), ('1/a', b'redirected once'), ('2/',), ('2/a', b'redirected twice'), ('3/',), ('3/a', b'redirected thrice'), ('4/',), ('4/a', b'redirected 4 times'), ('5/',), ('5/a', b'redirected 5 times')])

    def test_one_redirection(self):
        t = self.get_old_transport()
        new_prefix = 'http://{}:{}'.format(self.new_server.host, self.new_server.port)
        self.old_server.redirections = [('(.*)', '%s/1\\1' % new_prefix, 301)]
        self.assertEqual(b'redirected once', t.request('GET', t._remote_path('a'), retries=1).read())

    def test_five_redirections(self):
        t = self.get_old_transport()
        old_prefix = 'http://{}:{}'.format(self.old_server.host, self.old_server.port)
        new_prefix = 'http://{}:{}'.format(self.new_server.host, self.new_server.port)
        self.old_server.redirections = [('/1(.*)', '%s/2\\1' % old_prefix, 302), ('/2(.*)', '%s/3\\1' % old_prefix, 303), ('/3(.*)', '%s/4\\1' % old_prefix, 307), ('/4(.*)', '%s/5\\1' % new_prefix, 301), ('(/[^/]+)', '%s/1\\1' % old_prefix, 301)]
        self.assertEqual(b'redirected 5 times', t.request('GET', t._remote_path('a'), retries=6).read())
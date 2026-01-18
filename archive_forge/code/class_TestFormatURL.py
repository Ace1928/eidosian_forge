import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class TestFormatURL(test_utils.BaseTestCase):
    scenarios = [('transport', dict(transport='testtransport', virtual_host=None, hosts=[], expected='testtransport:///')), ('virtual_host', dict(transport='testtransport', virtual_host='/vhost', hosts=[], expected='testtransport:////vhost')), ('host', dict(transport='testtransport', virtual_host='/', hosts=[dict(hostname='host', port=10, username='bob', password='secret')], expected='testtransport://bob:secret@host:10//')), ('multi_host', dict(transport='testtransport', virtual_host='', hosts=[dict(hostname='h1', port=1000, username='b1', password='s1'), dict(hostname='h2', port=2000, username='b2', password='s2')], expected='testtransport://b1:s1@h1:1000,b2:s2@h2:2000/')), ('quoting', dict(transport='testtransport', virtual_host='/$', hosts=[dict(hostname='host', port=10, username='b$', password='s&')], expected='testtransport://b%24:s%26@host:10//%24'))]

    def test_parse_url(self):
        hosts = []
        for host in self.hosts:
            hosts.append(oslo_messaging.TransportHost(host.get('hostname'), host.get('port'), host.get('username'), host.get('password')))
        url = oslo_messaging.TransportURL(self.conf, self.transport, self.virtual_host, hosts)
        self.assertEqual(self.expected, str(url))
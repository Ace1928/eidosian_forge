from neutron_lib.api.definitions import dns
from neutron_lib.api.definitions import l3
from neutron_lib.tests.unit.api.definitions import base
class DnsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = dns
    extension_resources = (l3.FLOATINGIPS,)
    extension_attributes = (dns.DNSNAME, dns.DNSDOMAIN, dns.DNSASSIGNMENT)
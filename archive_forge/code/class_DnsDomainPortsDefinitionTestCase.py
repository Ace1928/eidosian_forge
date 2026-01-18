from neutron_lib.api.definitions import dns
from neutron_lib.api.definitions import dns_domain_ports
from neutron_lib.tests.unit.api.definitions import base
class DnsDomainPortsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = dns_domain_ports
    extension_attributes = (dns.DNSDOMAIN,)
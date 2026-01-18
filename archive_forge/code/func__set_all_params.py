import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallrule
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def _set_all_params(self, args={}):
    name = args.get('name') or 'my-name'
    description = args.get('description') or 'my-desc'
    source_ip = args.get('source_ip_address') or '192.168.1.0/24'
    destination_ip = args.get('destination_ip_address') or '192.168.2.0/24'
    source_port = args.get('source_port') or '0:65535'
    protocol = args.get('protocol') or 'udp'
    action = args.get('action') or 'deny'
    ip_version = args.get('ip_version') or '4'
    destination_port = args.get('destination_port') or '0:65535'
    destination_firewall_group = args.get('destination_firewall_group') or 'my-dst-fwg'
    source_firewall_group = args.get('source_firewall_group') or 'my-src-fwg'
    tenant_id = args.get('tenant_id') or 'my-tenant'
    arglist = ['--description', description, '--name', name, '--protocol', protocol, '--ip-version', ip_version, '--source-ip-address', source_ip, '--destination-ip-address', destination_ip, '--source-port', source_port, '--destination-port', destination_port, '--action', action, '--project', tenant_id, '--disable-rule', '--share', '--source-firewall-group', source_firewall_group, '--destination-firewall-group', destination_firewall_group]
    verifylist = [('name', name), ('description', description), ('share', True), ('protocol', protocol), ('ip_version', ip_version), ('source_ip_address', source_ip), ('destination_ip_address', destination_ip), ('source_port', source_port), ('destination_port', destination_port), ('action', action), ('disable_rule', True), ('project', tenant_id), ('source_firewall_group', source_firewall_group), ('destination_firewall_group', destination_firewall_group)]
    return (arglist, verifylist)
from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
class VirtualServerValidator(object):

    def __init__(self, module=None, client=None, want=None, have=None):
        self.have = have if have else ApiParameters()
        self.want = want if want else ModuleParameters()
        self.client = client
        self.module = module

    def check_update(self):
        self._override_port_by_type()
        self._override_protocol_by_type()
        self._verify_type_has_correct_profiles()
        self._verify_default_persistence_profile_for_type()
        self._verify_fallback_persistence_profile_for_type()
        self._update_persistence_profile()
        self._ensure_server_type_supports_vlans()
        self._verify_type_has_correct_ip_protocol()
        self._verify_dhcp_profile()
        self._verify_fastl4_profile()
        self._verify_stateless_profile()

    def check_create(self):
        self._set_default_ip_protocol()
        self._set_default_profiles()
        self._override_port_by_type()
        self._override_protocol_by_type()
        self._verify_type_has_correct_profiles()
        self._verify_default_persistence_profile_for_type()
        self._verify_fallback_persistence_profile_for_type()
        self._update_persistence_profile()
        self._verify_virtual_has_required_parameters()
        self._ensure_server_type_supports_vlans()
        self._override_vlans_if_all_specified()
        self._check_source_and_destination_match()
        self._verify_type_has_correct_ip_protocol()
        self._verify_minimum_profile()
        self._verify_dhcp_profile()
        self._verify_fastl4_profile()
        self._verify_stateless_profile_on_create()

    def _ensure_server_type_supports_vlans(self):
        """Verifies the specified server type supports VLANs

        A select number of server types do not support VLANs. This method
        checks to see if the specified types were provided along with VLANs.
        If they were, the module will raise an error informing the user that
        they need to either remove the VLANs, or, change the ``type``.

        Returns:
            None: Returned if no VLANs are specified.
        Raises:
            F5ModuleError: Raised if the server type conflicts with VLANs.
        """
        if self.want.enabled_vlans is None:
            return
        if self.want.type == 'internal':
            raise F5ModuleError("The 'internal' server type does not support VLANs.")

    def _override_vlans_if_all_specified(self):
        """Overrides any specified VLANs if "all" VLANs are specified

        The special setting "all VLANs" in a BIG-IP requires that no other VLANs
        be specified. If you specify any number of VLANs, AND include the "all"
        VLAN, this method will erase all of the other VLANs and only return the
        "all" VLAN.
        """
        all_vlans = ['/common/all', 'all']
        if self.want.enabled_vlans is not None:
            if any((x for x in self.want.enabled_vlans if x.lower() in all_vlans)):
                self.want.update(dict(enabled_vlans=[], vlans_disabled=True, vlans_enabled=False))

    def _override_port_by_type(self):
        if self.want.type == 'dhcp':
            self.want.update({'port': 67})
        elif self.want.type == 'internal':
            self.want.update({'port': 0})

    def _override_protocol_by_type(self):
        if self.want.type in ['stateless']:
            self.want.update({'ip_protocol': 17})

    def _check_source_and_destination_match(self):
        """Verify that destination and source are of the same IP version

        BIG-IP does not allow for mixing of the IP versions for destination and
        source addresses. For example, a destination IPv6 address cannot be
        associated with a source IPv4 address.

        This method checks that you specified the same IP version for these
        parameters.

        This method will not do this check if the virtual address name is used.

        Raises:
            F5ModuleError: Raised when the IP versions of source and destination differ.
        """
        if self.want.source and self.want.destination and (not self.want.destination_tuple.not_ip):
            want = ip_interface(u'{0}/{1}'.format(self.want.source_tuple.ip, self.want.source_tuple.cidr))
            have = ip_interface(u'{0}'.format(self.want.destination_tuple.ip))
            if want.version != have.version:
                raise F5ModuleError('The source and destination addresses for the virtual server must be be the same type (IPv4 or IPv6).')

    def _verify_type_has_correct_ip_protocol(self):
        if self.want.ip_protocol is None:
            return
        if self.want.type == 'standard':
            if self.want.ip_protocol not in [6, 17, 132, 51, 50, 'any']:
                raise F5ModuleError("The 'standard' server type does not support the specified 'ip_protocol'.")
        elif self.want.type == 'performance-http':
            if self.want.ip_protocol not in [6]:
                raise F5ModuleError("The 'performance-http' server type does not support the specified 'ip_protocol'.")
        elif self.want.type == 'stateless':
            if self.want.ip_protocol not in [17]:
                raise F5ModuleError("The 'stateless' server type does not support the specified 'ip_protocol'.")
        elif self.want.type == 'dhcp':
            if self.want.ip_protocol is not None:
                raise F5ModuleError("The 'dhcp' server type does not support an 'ip_protocol'.")
        elif self.want.type == 'internal':
            if self.want.ip_protocol not in [6, 17]:
                raise F5ModuleError("The 'internal' server type does not support the specified 'ip_protocol'.")
        elif self.want.type == 'message-routing':
            if self.want.ip_protocol not in [6, 17, 132, 'all', 'any']:
                raise F5ModuleError("The 'message-routing' server type does not support the specified 'ip_protocol'.")

    def _verify_virtual_has_required_parameters(self):
        """Verify that the virtual has required parameters

        Virtual servers require several parameters that are not necessarily required
        when updating the virtual. This method will check for the required params
        upon creation.

        Ansible supports ``default`` variables in an Argument Spec, but those defaults
        apply to all operations; including create, update, and delete. Since users are not
        required to always specify these parameters, we cannot use Ansible's facility.
        If we did, and then users would be required to provide them when, for example,
        they attempted to delete a virtual (even though they are not required to delete
        a virtual.

        Raises:
             F5ModuleError: Raised when the user did not specify required parameters.
        """
        required_resources = ['destination', 'port']
        if self.want.type == 'internal':
            return
        if all((getattr(self.want, v) is None for v in required_resources)):
            raise F5ModuleError('You must specify both of ' + ', '.join(required_resources))

    def _verify_default_persistence_profile_for_type(self):
        """Verify that the server type supports default persistence profiles

        Verifies that the specified server type supports default persistence profiles.
        Some virtual servers do not support these types of profiles. This method will
        check that the type actually supports what you are sending it.

        Types that do not, at this time, support default persistence profiles include,

        * dhcp
        * message-routing
        * reject
        * stateless
        * forwarding-ip
        * forwarding-l2

        Raises:
            F5ModuleError: Raised if server type does not support default persistence profiles.
        """
        default_profile_not_allowed = ['dhcp', 'message-routing', 'reject', 'stateless', 'forwarding-ip', 'forwarding-l2']
        if self.want.ip_protocol in default_profile_not_allowed:
            raise F5ModuleError("The '{0}' server type does not support a 'default_persistence_profile'".format(self.want.type))

    def _verify_fallback_persistence_profile_for_type(self):
        """Verify that the server type supports fallback persistence profiles

        Verifies that the specified server type supports fallback persistence profiles.
        Some virtual servers do not support these types of profiles. This method will
        check that the type actually supports what you are sending it.

        Types that do not, at this time, support fallback persistence profiles include,

        * dhcp
        * message-routing
        * reject
        * stateless
        * forwarding-ip
        * forwarding-l2
        * performance-http

        Raises:
            F5ModuleError: Raised if server type does not support fallback persistence profiles.
        """
        default_profile_not_allowed = ['dhcp', 'message-routing', 'reject', 'stateless', 'forwarding-ip', 'forwarding-l2', 'performance-http']
        if self.want.ip_protocol in default_profile_not_allowed:
            raise F5ModuleError("The '{0}' server type does not support a 'fallback_persistence_profile'".format(self.want.type))

    def _update_persistence_profile(self):
        if self.want.default_persistence_profile is not None:
            self.want.update({'default_persistence_profile': self.want.default_persistence_profile})

    def _verify_type_has_correct_profiles(self):
        """Verify that specified server type does not include forbidden profiles

        The type of the server determines the ``type``s of profiles that it accepts. This
        method checks that the server ``type`` that you specified is indeed one that can
        accept the profiles that you specified.

        The common situations are

        * ``standard`` types that include ``fasthttp``, ``fastl4``, or ``message routing`` profiles
        * ``fasthttp`` types that are missing a ``fasthttp`` profile
        * ``fastl4`` types that are missing a ``fastl4`` profile
        * ``message-routing`` types that are missing ``diameter`` or ``sip`` profiles

        Raises:
            F5ModuleError: Raised when a validation check fails.
        """
        if self.want.type == 'standard':
            if self.want.has_fasthttp_profiles:
                raise F5ModuleError("A 'standard' type may not have 'fasthttp' profiles.")
            if self.want.has_fastl4_profiles:
                raise F5ModuleError("A 'standard' type may not have 'fastl4' profiles.")
        elif self.want.type == 'performance-http':
            if not self.want.has_fasthttp_profiles:
                raise F5ModuleError("A 'fasthttp' type must have at least one 'fasthttp' profile.")
        elif self.want.type == 'performance-l4':
            if not self.want.has_fastl4_profiles:
                raise F5ModuleError("A 'fastl4' type must have at least one 'fastl4' profile.")
        elif self.want.type == 'message-routing':
            if not self.want.has_message_routing_profiles:
                raise F5ModuleError("A 'message-routing' type must have either a 'sip' or 'diameter' profile.")

    def _set_default_ip_protocol(self):
        if self.want.type == 'dhcp':
            return
        if self.want.ip_protocol is None:
            self.want.update({'ip_protocol': 6})

    def _set_default_profiles(self):
        if self.want.type == 'standard':
            if not self.want.profiles:
                if self.want.ip_protocol == 6:
                    self.want.update({'profiles': ['tcp']})
                if self.want.ip_protocol == 17:
                    self.want.update({'profiles': ['udp']})
                if self.want.ip_protocol == 132:
                    self.want.update({'profiles': ['sctp']})

    def _verify_minimum_profile(self):
        if self.want.profiles:
            return None
        if self.want.type == 'internal' and self.want.profiles == '':
            raise F5ModuleError("An 'internal' server must have at least one profile relevant to its 'ip_protocol'. For example, 'tcp', 'udp', or variations of those.")

    def _verify_dhcp_profile(self):
        if self.want.type != 'dhcp':
            return
        if self.want.profiles is None:
            return
        have = set(self.read_dhcp_profiles_from_device())
        want = set([x['fullPath'] for x in self.want.profiles])
        if have.intersection(want):
            return True
        raise F5ModuleError("A dhcp profile, such as 'dhcpv4', or 'dhcpv6' must be specified when 'type' is 'dhcp'.")

    def _verify_fastl4_profile(self):
        if self.want.type != 'performance-l4':
            return
        if self.want.profiles is None:
            return
        have = set(self.read_fastl4_profiles_from_device())
        want = set([x['fullPath'] for x in self.want.profiles])
        if have.intersection(want):
            return True
        raise F5ModuleError("A performance-l4 profile, such as 'fastL4', must be specified when 'type' is 'performance-l4'.")

    def _verify_fasthttp_profile(self):
        if self.want.type != 'performance-http':
            return
        if self.want.profiles is None:
            return
        have = set(self.read_fasthttp_profiles_from_device())
        want = set([x['fullPath'] for x in self.want.profiles])
        if have.intersection(want):
            return True
        raise F5ModuleError("A performance-http profile, such as 'fasthttp', must be specified when 'type' is 'performance-http'.")

    def _verify_stateless_profile_on_create(self):
        if self.want.type != 'stateless':
            return
        result = self._verify_stateless_profile()
        if result is None:
            raise F5ModuleError("A udp profile, must be specified when 'type' is 'stateless'.")

    def _verify_stateless_profile(self):
        if self.want.type != 'stateless':
            return
        if self.want.profiles is None:
            return
        have = set(self.read_udp_profiles_from_device())
        want = set([x['fullPath'] for x in self.want.profiles])
        if have.intersection(want):
            return True
        raise F5ModuleError("A udp profile, must be specified when 'type' is 'stateless'.")

    def read_dhcp_profiles_from_device(self):
        result = []
        result += self.read_dhcpv4_profiles_from_device()
        result += self.read_dhcpv6_profiles_from_device()
        return result

    def read_dhcpv4_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/dhcpv4/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        result = [fq_name(self.want.partition, x['name']) for x in response['items']]
        return result

    def read_dhcpv6_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/dhcpv6/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        result = [fq_name(self.want.partition, x['name']) for x in response['items']]
        return result

    def read_fastl4_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/fastl4/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        result = [x['fullPath'] for x in response['items']]
        return result

    def read_fasthttp_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/fasthttp/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        result = [fq_name(self.want.partition, x['name']) for x in response['items']]
        return result

    def read_udp_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/udp/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        result = [fq_name(self.want.partition, x['name']) for x in response['items']]
        return result
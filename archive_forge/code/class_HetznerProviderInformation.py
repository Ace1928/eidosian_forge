from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
class HetznerProviderInformation(ProviderInformation):

    def get_supported_record_types(self):
        """
        Return a list of supported record types.
        """
        return ['A', 'AAAA', 'NS', 'MX', 'CNAME', 'RP', 'TXT', 'SOA', 'HINFO', 'SRV', 'DANE', 'TLSA', 'DS', 'CAA']

    def get_zone_id_type(self):
        """
        Return the (short) type for zone IDs, like ``'int'`` or ``'str'``.
        """
        return 'str'

    def get_record_id_type(self):
        """
        Return the (short) type for record IDs, like ``'int'`` or ``'str'``.
        """
        return 'str'

    def get_record_default_ttl(self):
        """
        Return the default TTL for records, like 300, 3600 or None.
        None means that some other TTL (usually from the zone) will be used.
        """
        return None

    def normalize_prefix(self, prefix):
        """
        Given a prefix (string or None), return its normalized form.

        The result should always be None for the trivial prefix, and a non-zero length DNS name
        for a non-trivial prefix.

        If a provider supports other identifiers for the trivial prefix, such as '@', this
        function needs to convert them to None as well.
        """
        return None if prefix in ('@', '') else prefix

    def supports_bulk_actions(self):
        """
        Return whether the API supports some kind of bulk actions.
        """
        return True

    def txt_record_handling(self):
        """
        Return how the API handles TXT records.

        Returns one of the following strings:
        * 'decoded' - the API works with unencoded values
        * 'encoded' - the API works with encoded values
        * 'encoded-no-char-encoding' - the API works with encoded values, but without character encoding
        """
        return 'encoded-no-char-encoding'
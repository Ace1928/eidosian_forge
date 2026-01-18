from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible.module_utils.common.validation import (
@six.add_metaclass(abc.ABCMeta)
class ProviderInformation(object):

    @abc.abstractmethod
    def get_zone_id_type(self):
        """
        Return the (short) type for zone IDs, like ``'int'`` or ``'str'``.
        """

    @abc.abstractmethod
    def get_record_id_type(self):
        """
        Return the (short) type for record IDs, like ``'int'`` or ``'str'``.
        """

    @abc.abstractmethod
    def get_record_default_ttl(self):
        """
        Return the default TTL for records, like 300, 3600 or None.
        None means that some other TTL (usually from the zone) will be used.
        """

    @abc.abstractmethod
    def get_supported_record_types(self):
        """
        Return a list of supported record types.
        """

    def normalize_prefix(self, prefix):
        """
        Given a prefix (string or None), return its normalized form.

        The result should always be None for the trivial prefix, and a non-zero length DNS name
        for a non-trivial prefix.

        If a provider supports other identifiers for the trivial prefix, such as '@', this
        function needs to convert them to None as well.
        """
        return prefix or None

    def supports_bulk_actions(self):
        """
        Return whether the API supports some kind of bulk actions.
        """
        return False

    @abc.abstractmethod
    def txt_record_handling(self):
        """
        Return how the API handles TXT records.

        Returns one of the following strings:
        * 'decoded' - the API works with unencoded values
        * 'encoded' - the API works with encoded values
        * 'encoded-no-char-encoding' - the API works with encoded values, but without character encoding
        """

    def txt_character_encoding(self):
        """
        Return how the API handles escape sequences in TXT records.

        Returns one of the following strings:
        * 'octal' - the API works with octal escape sequences
        * 'decimal' - the API works with decimal escape sequences

        This return value is only used if txt_record_handling returns 'encoded'.

        WARNING: the default return value will change to 'decimal' for community.dns 3.0.0!
        """
        return 'octal'
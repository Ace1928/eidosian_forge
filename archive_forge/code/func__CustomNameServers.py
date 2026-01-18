from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.dns import util as dns_api_util
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import six
def _CustomNameServers(domains_messages, name_servers, ds_records=None):
    """Validates name servers and returns (dns_settings, update_mask)."""
    if not ds_records:
        ds_records = []
    normalized_name_servers = list(map(util.NormalizeDomainName, name_servers))
    for ns, normalized in zip(name_servers, normalized_name_servers):
        if not util.ValidateDomainName(normalized):
            raise exceptions.Error("Invalid name server: '{}'.".format(ns))
    update_mask = DnsUpdateMask(name_servers=True, custom_dnssec=True)
    dns_settings = domains_messages.DnsSettings(customDns=domains_messages.CustomDns(nameServers=normalized_name_servers, dsRecords=ds_records))
    return (dns_settings, update_mask)
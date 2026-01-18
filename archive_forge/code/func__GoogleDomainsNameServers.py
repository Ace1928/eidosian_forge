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
def _GoogleDomainsNameServers(domains_messages, enable_dnssec):
    """Enable Google Domains name servers and returns (dns_settings, update_mask)."""
    update_mask = DnsUpdateMask(name_servers=True, google_domains_dnssec=True)
    ds_state = domains_messages.GoogleDomainsDns.DsStateValueValuesEnum.DS_RECORDS_PUBLISHED
    if not enable_dnssec:
        ds_state = domains_messages.GoogleDomainsDns.DsStateValueValuesEnum.DS_RECORDS_UNPUBLISHED
    dns_settings = domains_messages.DnsSettings(googleDomainsDns=domains_messages.GoogleDomainsDns(dsState=ds_state))
    return (dns_settings, update_mask)
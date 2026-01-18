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
def _ParseDnsSettingsFromFile(domains_messages, path):
    """Parses dns_settings from a yaml file.

  Args:
    domains_messages: Cloud Domains messages module.
    path: YAML file path.

  Returns:
    Pair (DnsSettings, DnsUpdateMask) or (None, None) if path is None.
  """
    dns_settings = util.ParseMessageFromYamlFile(path, domains_messages.DnsSettings, "DNS settings file '{}' does not contain valid dns_settings message".format(path))
    if not dns_settings:
        return (None, None)
    update_mask = None
    if dns_settings.googleDomainsDns is not None:
        update_mask = DnsUpdateMask(name_servers=True, google_domains_dnssec=True, glue_records=True)
    elif dns_settings.customDns is not None:
        update_mask = DnsUpdateMask(name_servers=True, custom_dnssec=True, glue_records=True)
    else:
        raise exceptions.Error("dnsProvider is not present in DNS settings file '{}'.".format(path))
    return (dns_settings, update_mask)
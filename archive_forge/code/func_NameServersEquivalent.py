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
def NameServersEquivalent(prev_dns_settings, new_dns_settings):
    """Checks if dns settings have equivalent name servers."""
    if prev_dns_settings.googleDomainsDns:
        return bool(new_dns_settings.googleDomainsDns)
    if prev_dns_settings.customDns:
        if not new_dns_settings.customDns:
            return False
        prev_ns = sorted(map(util.NormalizeDomainName, prev_dns_settings.customDns.nameServers))
        new_ns = sorted(map(util.NormalizeDomainName, new_dns_settings.customDns.nameServers))
        return prev_ns == new_ns
    return False
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
def PromptForNameServers(api_version, domain, enable_dnssec=None, dns_settings=None, print_format='default'):
    """Asks the user to provide DNS settings interactively.

  Args:
    api_version: Cloud Domains API version to call.
    domain: Domain name corresponding to the DNS settings.
    enable_dnssec: Should the DNSSEC be enabled.
    dns_settings: Current DNS configuration (or None if resource is not yet
      created).
    print_format: Print format to use when showing current dns_settings.

  Returns:
    A pair: (messages.DnsSettings, DnsUpdateMask) to be updated, or (None, None)
    if the user cancelled.
  """
    domains_messages = registrations.GetMessagesModule(api_version)
    options = ['Provide name servers list', 'Provide Cloud DNS Managed Zone name', 'Use free name servers provided by Google Domains']
    if dns_settings is not None:
        log.status.Print('Your current DNS settings are:')
        resource_printer.Print(dns_settings, print_format, out=sys.stderr)
        message = 'You can provide your DNS settings by specifying name servers, a Cloud DNS Managed Zone name or by choosing free name servers provided by Google Domains'
        cancel_option = True
        default = len(options)
    else:
        options = options[:2]
        message = 'You can provide your DNS settings by specifying name servers or a Cloud DNS Managed Zone name'
        cancel_option = False
        default = 1
    index = console_io.PromptChoice(message=message, options=options, cancel_option=cancel_option, default=default)
    name_servers = []
    if index == 0:
        while len(name_servers) < 2:
            while True:
                ns = console_io.PromptResponse('Name server (empty line to finish):  ')
                if not ns:
                    break
                if not util.ValidateDomainName(ns):
                    log.status.Print("Invalid name server: '{}'.".format(ns))
                else:
                    name_servers += [ns]
            if len(name_servers) < 2:
                log.status.Print('You have to provide at least 2 name servers.')
        return _CustomNameServers(domains_messages, name_servers)
    elif index == 1:
        while True:
            zone = util.PromptWithValidator(validator=util.ValidateNonEmpty, error_message=' Cloud DNS Managed Zone name must not be empty.', prompt_string='Cloud DNS Managed Zone name:  ')
            try:
                name_servers, ds_records = _GetCloudDnsDetails(domains_messages, zone, domain, enable_dnssec)
            except (exceptions.Error, calliope_exceptions.HttpException) as e:
                log.status.Print(six.text_type(e))
            else:
                break
        return _CustomNameServers(domains_messages, name_servers, ds_records)
    elif index == 2:
        return _GoogleDomainsNameServers(domains_messages, enable_dnssec)
    else:
        return (None, None)
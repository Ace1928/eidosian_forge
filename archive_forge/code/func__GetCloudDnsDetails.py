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
def _GetCloudDnsDetails(domains_messages, cloud_dns_zone, domain, enable_dnssec):
    """Fetches list of name servers from provided Cloud DNS Managed Zone.

  Args:
    domains_messages: Cloud Domains messages module.
    cloud_dns_zone: Cloud DNS Zone resource reference.
    domain: Domain name.
    enable_dnssec: If true, try to read DNSSEC information from the Zone.

  Returns:
    A pair: List of name servers and a list of Ds records (or [] if e.g. the
    Zone is not signed).
  """
    dns_api_version = 'v1'
    dns = apis.GetClientInstance('dns', dns_api_version)
    dns_messages = dns.MESSAGES_MODULE
    zone_ref = dns_api_util.GetRegistry(dns_api_version).Parse(cloud_dns_zone, params={'project': properties.VALUES.core.project.GetOrFail}, collection='dns.managedZones')
    try:
        zone = dns.managedZones.Get(dns_messages.DnsManagedZonesGetRequest(project=zone_ref.project, managedZone=zone_ref.managedZone))
    except apitools_exceptions.HttpError as error:
        raise calliope_exceptions.HttpException(error)
    domain_with_dot = domain + '.'
    if zone.dnsName != domain_with_dot:
        raise exceptions.Error("The dnsName '{}' of specified Cloud DNS zone '{}' does not match the registration domain '{}'".format(zone.dnsName, cloud_dns_zone, domain_with_dot))
    if zone.visibility != dns_messages.ManagedZone.VisibilityValueValuesEnum.public:
        raise exceptions.Error("Cloud DNS Zone '{}' is not public.".format(cloud_dns_zone))
    if not enable_dnssec:
        return (zone.nameServers, [])
    signed = dns_messages.ManagedZoneDnsSecConfig.StateValueValuesEnum.on
    if not zone.dnssecConfig or zone.dnssecConfig.state != signed:
        log.status.Print("Cloud DNS Zone '{}' is not signed. DNSSEC won't be enabled.".format(cloud_dns_zone))
        return (zone.nameServers, [])
    try:
        dns_keys = []
        req = dns_messages.DnsDnsKeysListRequest(project=zone_ref.project, managedZone=zone_ref.managedZone)
        while True:
            resp = dns.dnsKeys.List(req)
            dns_keys += resp.dnsKeys
            req.pageToken = resp.nextPageToken
            if not resp.nextPageToken:
                break
    except apitools_exceptions.HttpError as error:
        log.status.Print("Cannot read DS records from Cloud DNS Zone '{}': {}. DNSSEC won't be enabled.".format(cloud_dns_zone, error))
    ds_records = _ConvertDnsKeys(domains_messages, dns_messages, dns_keys)
    if not ds_records:
        log.status.Print("No supported DS records found in Cloud DNS Zone '{}'. DNSSEC won't be enabled.".format(cloud_dns_zone))
        return (zone.nameServers, [])
    return (zone.nameServers, ds_records)
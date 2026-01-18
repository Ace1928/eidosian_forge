from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def ValidateServiceProxyFlags(args):
    """Validates the values of all --service-proxy related flags."""
    if getattr(args, 'service_proxy', False):
        if args.no_scopes:
            raise exceptions.ConflictingArgumentsException('--service-proxy', '--no-scopes')
        if 'serving-ports' in args.service_proxy:
            try:
                serving_ports = list(map(int, args.service_proxy['serving-ports'].split(';')))
                for port in serving_ports:
                    if port < 1 or port > 65535:
                        raise ValueError
            except ValueError:
                raise exceptions.InvalidArgumentException('serving-ports', 'List of ports can only contain numbers between 1 and 65535.')
        if 'proxy-port' in args.service_proxy:
            try:
                proxy_port = args.service_proxy['proxy-port']
                if proxy_port < 1025 or proxy_port > 65535:
                    raise ValueError
            except ValueError:
                raise exceptions.InvalidArgumentException('proxy-port', 'Port value can only be between 1025 and 65535.')
        if 'exclude-outbound-ip-ranges' in args.service_proxy:
            if 'intercept-all-outbound-traffic' not in args.service_proxy:
                raise exceptions.RequiredArgumentException('intercept-all-outbound-traffic', 'exclude-outbound-ip-ranges parameters requires intercept-all-outbound-traffic to be set')
            ip_ranges = args.service_proxy['exclude-outbound-ip-ranges'].split(';')
            for ip_range in ip_ranges:
                try:
                    ipaddress.ip_network(ip_range)
                except ValueError:
                    raise exceptions.InvalidArgumentException('exclude-outbound-ip-ranges', 'List of IPs may contain only IPs & CIDRs.')
        if 'exclude-outbound-port-ranges' in args.service_proxy:
            if 'intercept-all-outbound-traffic' not in args.service_proxy:
                raise exceptions.RequiredArgumentException('intercept-all-outbound-traffic', 'exclude-outbound-port-ranges parameters requires intercept-all-outbound-traffic to be set')
            port_ranges = args.service_proxy['exclude-outbound-port-ranges'].split(';')
            for port_range in port_ranges:
                ports = port_range.split('-')
                try:
                    if len(ports) == 1:
                        ValidateSinglePort(ports[0])
                    elif len(ports) == 2:
                        ValidateSinglePort(ports[0])
                        ValidateSinglePort(ports[1])
                    else:
                        raise ValueError
                except ValueError:
                    raise exceptions.InvalidArgumentException('exclude-outbound-port-ranges', 'List of port ranges can only contain numbers between 1 and 65535, i.e. "80;8080-8090".')
        if 'scope' in args.service_proxy and 'mesh' in args.service_proxy:
            raise exceptions.ConflictingArgumentsException('--service-proxy:scope', '--service-proxy:mesh')
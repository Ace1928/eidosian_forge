from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def ModifyToAddInterface(self, router_ref, args, existing):
    """Mutate the router to add an interface."""
    replacement = encoding.CopyProtoMessage(existing)
    new_interface = self._messages.Interface(name=args.interface_name)
    if args.interconnect_attachment is not None:
        attachment_ref = self._resource_parser.Create('edgenetwork.projects.locations.zones.interconnectAttachments', interconnectAttachmentsId=args.interconnect_attachment, projectsId=router_ref.projectsId, locationsId=router_ref.locationsId, zonesId=router_ref.zonesId)
        if args.ip_mask_length is None or args.ip_address is None:
            raise parser_errors.ArgumentException('--ip-address and --ip-mask-length must be set')
        try:
            ip_address = ipaddress.ip_address(args.ip_address)
        except ValueError as err:
            raise parser_errors.ArgumentException(str(err))
        if args.ip_mask_length > ip_address.max_prefixlen:
            raise parser_errors.ArgumentException('--ip-mask-length should be less than %s' % ip_address.max_prefixlen)
        cidr = '{0}/{1}'.format(args.ip_address, args.ip_mask_length)
        if ip_address.version == 4:
            new_interface.ipv4Cidr = cidr
        else:
            new_interface.ipv6Cidr = cidr
        new_interface.linkedInterconnectAttachment = attachment_ref.RelativeName()
    if args.subnetwork is not None:
        subnet_ref = self._resource_parser.Create('edgenetwork.projects.locations.zones.subnets', subnetsId=args.subnetwork, projectsId=router_ref.projectsId, locationsId=router_ref.locationsId, zonesId=router_ref.zonesId)
        new_interface.subnetwork = subnet_ref.RelativeName()
    if args.loopback_ip_addresses is not None:
        new_interface.loopbackIpAddresses = args.loopback_ip_addresses
    replacement.interface.append(new_interface)
    return replacement
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AddAccessConfigInstances(base.SilentCommand):
    """Create a Compute Engine virtual machine access configuration."""
    _support_public_dns = False

    @classmethod
    def Args(cls, parser):
        _Args(parser, support_public_dns=cls._support_public_dns)

    def Run(self, args):
        """Invokes request necessary for adding an access config."""
        flags.ValidateNetworkTierArgs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))
        access_config = client.messages.AccessConfig(name=args.access_config_name, natIP=args.address, type=client.messages.AccessConfig.TypeValueValuesEnum.ONE_TO_ONE_NAT)
        if self._support_public_dns:
            flags.ValidatePublicDnsFlags(args)
            if args.no_public_dns is True:
                access_config.setPublicDns = False
            elif args.public_dns is True:
                access_config.setPublicDns = True
        flags.ValidatePublicPtrFlags(args)
        if args.no_public_ptr is True:
            access_config.setPublicPtr = False
        elif args.public_ptr is True:
            access_config.setPublicPtr = True
        if args.no_public_ptr_domain is not True and args.public_ptr_domain is not None:
            access_config.publicPtrDomainName = args.public_ptr_domain
        network_tier = getattr(args, 'network_tier', None)
        if network_tier is not None:
            access_config.networkTier = client.messages.AccessConfig.NetworkTierValueValuesEnum(network_tier)
        request = client.messages.ComputeInstancesAddAccessConfigRequest(accessConfig=access_config, instance=instance_ref.Name(), networkInterface=args.network_interface, project=instance_ref.project, zone=instance_ref.zone)
        return client.MakeRequests([(client.apitools_client.instances, 'AddAccessConfig', request)])
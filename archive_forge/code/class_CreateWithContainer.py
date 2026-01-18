from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class CreateWithContainer(base.CreateCommand):
    """Command for creating VM instance templates hosting Docker images."""

    @staticmethod
    def Args(parser):
        _Args(parser, base.ReleaseTrack.GA, container_mount_enabled=True, support_multi_writer=False, support_region_instance_template=False)
        instances_flags.AddPrivateIpv6GoogleAccessArgForTemplate(parser, utils.COMPUTE_GA_API_VERSION)

    def _ValidateArgs(self, args):
        instances_flags.ValidateKonletArgs(args)
        instances_flags.ValidateDiskCommonFlags(args)
        instances_flags.ValidateServiceAccountAndScopeArgs(args)
        instances_flags.ValidateNicFlags(args)
        if instance_utils.UseExistingBootDisk(args.disk or []):
            raise exceptions.InvalidArgumentException('--disk', 'Boot disk specified for containerized VM.')

    def _GetBootDiskSize(self, args):
        boot_disk_size_gb = utils.BytesToGb(args.boot_disk_size)
        utils.WarnIfDiskSizeIsTooSmall(boot_disk_size_gb, args.boot_disk_type)
        return boot_disk_size_gb

    def _GetInstanceTemplateRef(self, args, holder):
        return CreateWithContainer.InstanceTemplateArg.ResolveAsResource(args, holder.resources)

    def _GetUserMetadata(self, args, client, instance_template_ref, container_mount_disk_enabled=False, container_mount_disk=None):
        user_metadata = instance_utils.GetValidatedMetadata(args, client)
        return containers_utils.CreateKonletMetadataMessage(client.messages, args, instance_template_ref.Name(), user_metadata, container_mount_disk_enabled, container_mount_disk)

    def _GetNetworkInterfaces(self, args, client, holder):
        if args.network_interface:
            return instance_template_utils.CreateNetworkInterfaceMessages(resources=holder.resources, scope_lister=flags.GetDefaultScopeLister(client), messages=client.messages, network_interface_arg=args.network_interface, subnet_region=args.region)
        stack_type = getattr(args, 'stack_type', None)
        ipv6_network_tier = getattr(args, 'ipv6_network_tier', None)
        ipv6_address = getattr(args, 'ipv6_address', None)
        ipv6_prefix_length = getattr(args, 'ipv6_prefix_length', None)
        external_ipv6_address = getattr(args, 'external_ipv6_address', None)
        external_ipv6_prefix_length = getattr(args, 'external_ipv6_prefix_length', None)
        internal_ipv6_address = getattr(args, 'internal_ipv6_address', None)
        internal_ipv6_prefix_length = getattr(args, 'internal_ipv6_prefix_length', None)
        return [instance_template_utils.CreateNetworkInterfaceMessage(resources=holder.resources, scope_lister=flags.GetDefaultScopeLister(client), messages=client.messages, network=args.network, private_ip=args.private_network_ip, subnet_region=args.region, subnet=args.subnet, address=instance_template_utils.EPHEMERAL_ADDRESS if not args.no_address and (not args.address) else args.address, network_tier=getattr(args, 'network_tier', None), stack_type=stack_type, ipv6_network_tier=ipv6_network_tier, ipv6_address=ipv6_address, ipv6_prefix_length=ipv6_prefix_length, external_ipv6_address=external_ipv6_address, external_ipv6_prefix_length=external_ipv6_prefix_length, internal_ipv6_address=internal_ipv6_address, internal_ipv6_prefix_length=internal_ipv6_prefix_length)]

    def _GetScheduling(self, args, client):
        return instance_utils.CreateSchedulingMessage(messages=client.messages, maintenance_policy=args.maintenance_policy, preemptible=args.preemptible, restart_on_failure=args.restart_on_failure)

    def _GetServiceAccounts(self, args, client):
        if args.no_service_account:
            service_account = None
        else:
            service_account = args.service_account
        return instance_utils.CreateServiceAccountMessages(messages=client.messages, scopes=[] if args.no_scopes else args.scopes, service_account=service_account)

    def _GetImageUri(self, args, client, holder, instance_template_ref):
        if args.IsSpecified('image') or args.IsSpecified('image_family') or args.IsSpecified('image_project'):
            image_expander = image_utils.ImageExpander(client, holder.resources)
            image_uri, _ = image_expander.ExpandImageFlag(user_project=instance_template_ref.project, image=args.image, image_family=args.image_family, image_project=args.image_project)
            if holder.resources.Parse(image_uri).project != 'cos-cloud':
                log.warning('This container deployment mechanism requires a Container-Optimized OS image in order to work. Select an image from a cos-cloud project (cos-stable, cos-beta, cos-dev image families).')
        else:
            image_uri = containers_utils.ExpandKonletCosImageFlag(client)
        return image_uri

    def _GetMachineType(self, args):
        return instance_utils.InterpretMachineType(machine_type=args.machine_type, custom_cpu=args.custom_cpu, custom_memory=args.custom_memory, ext=getattr(args, 'custom_extensions', None), vm_type=getattr(args, 'custom_vm_type', None))

    def _GetDisks(self, args, client, holder, instance_template_ref, image_uri, match_container_mount_disks=False):
        boot_disk_size_gb = self._GetBootDiskSize(args)
        create_boot_disk = not instance_utils.UseExistingBootDisk((args.disk or []) + (args.create_disk or []))
        return instance_template_utils.CreateDiskMessages(args, client, holder.resources, instance_template_ref.project, image_uri, boot_disk_size_gb, create_boot_disk=create_boot_disk, match_container_mount_disks=match_container_mount_disks)

    def Run(self, args):
        """Issues an InstanceTemplates.Insert request.

    Args:
      args: the argparse arguments that this command was invoked with.

    Returns:
      an InstanceTemplate message object
    """
        self._ValidateArgs(args)
        instances_flags.ValidateNetworkTierArgs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        container_mount_disk = instances_flags.GetValidatedContainerMountDisk(holder, args.container_mount_disk, args.disk, args.create_disk)
        client = holder.client
        instance_template_ref = self._GetInstanceTemplateRef(args, holder)
        image_uri = self._GetImageUri(args, client, holder, instance_template_ref)
        labels = containers_utils.GetLabelsMessageWithCosVersion(None, image_uri, holder.resources, client.messages.InstanceProperties)
        argument_labels = labels_util.ParseCreateArgs(args, client.messages.InstanceProperties.LabelsValue)
        if argument_labels:
            labels.additionalProperties.extend(argument_labels.additionalProperties)
        metadata = self._GetUserMetadata(args, client, instance_template_ref, container_mount_disk_enabled=True, container_mount_disk=container_mount_disk)
        network_interfaces = self._GetNetworkInterfaces(args, client, holder)
        scheduling = self._GetScheduling(args, client)
        service_accounts = self._GetServiceAccounts(args, client)
        machine_type = self._GetMachineType(args)
        disks = self._GetDisks(args, client, holder, instance_template_ref, image_uri, match_container_mount_disks=True)
        confidential_instance_config = create_utils.BuildConfidentialInstanceConfigMessage(messages=client.messages, args=args, support_confidential_compute_type=False, support_confidential_compute_type_tdx=False)
        properties = client.messages.InstanceProperties(machineType=machine_type, disks=disks, canIpForward=args.can_ip_forward, confidentialInstanceConfig=confidential_instance_config, labels=labels, metadata=metadata, minCpuPlatform=args.min_cpu_platform, networkInterfaces=network_interfaces, serviceAccounts=service_accounts, scheduling=scheduling, tags=containers_utils.CreateTagsMessage(client.messages, args.tags))
        if args.private_ipv6_google_access_type is not None:
            properties.privateIpv6GoogleAccess = instances_flags.GetPrivateIpv6GoogleAccessTypeFlagMapperForTemplate(client.messages).GetEnumForChoice(args.private_ipv6_google_access_type)
        shielded_instance_config_message = create_utils.BuildShieldedInstanceConfigMessage(messages=client.messages, args=args)
        if shielded_instance_config_message:
            properties.shieldedInstanceConfig = shielded_instance_config_message
        request = client.messages.ComputeInstanceTemplatesInsertRequest(instanceTemplate=client.messages.InstanceTemplate(properties=properties, description=args.description, name=instance_template_ref.Name()), project=instance_template_ref.project)
        return client.MakeRequests([(client.apitools_client.instanceTemplates, 'Insert', request)])
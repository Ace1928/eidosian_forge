from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
class InstancesRemoveMetadata(base.UpdateCommand):
    """Remove instance metadata.
  """

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser, operation_type='set metadata on')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--all', action='store_true', default=False, help='If provided, all metadata entries are removed.')
        group.add_argument('--keys', type=arg_parsers.ArgList(min_length=1), metavar='KEY', help='The keys of the entries to remove.')

    def CreateReference(self, client, resources, args):
        return flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=flags.GetInstanceZoneScopeLister(client))

    def GetGetRequest(self, client, instance_ref):
        return (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))

    def GetSetRequest(self, client, instance_ref, replacement):
        return (client.apitools_client.instances, 'SetMetadata', client.messages.ComputeInstancesSetMetadataRequest(metadata=replacement.metadata, **instance_ref.AsDict()))

    def Modify(self, client, args, existing):
        new_object = encoding.CopyProtoMessage(existing)
        existing_metadata = getattr(existing, 'metadata', None)
        new_object.metadata = metadata_utils.RemoveEntries(client.messages, existing_metadata=existing_metadata, keys=args.keys, remove_all=args.all)
        if metadata_utils.MetadataEqual(existing_metadata, new_object.metadata):
            return None
        else:
            return new_object

    def Run(self, args):
        if not args.all and (not args.keys):
            raise calliope_exceptions.OneOfArgumentsRequiredException(['--keys', '--all'], 'One of [--all] or [--keys] must be provided.')
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = self.CreateReference(client, holder.resources, args)
        get_request = self.GetGetRequest(client, instance_ref)
        objects = client.MakeRequests([get_request])
        new_object = self.Modify(client, args, objects[0])
        if not new_object or objects[0] == new_object:
            log.status.Print('No change requested; skipping update for [{0}].'.format(objects[0].name))
            return objects
        return client.MakeRequests([self.GetSetRequest(client, instance_ref, new_object)])
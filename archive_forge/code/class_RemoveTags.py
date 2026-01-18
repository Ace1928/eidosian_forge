from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
class RemoveTags(base.UpdateCommand):
    """Remove tags from Compute Engine virtual machine instances."""

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser, operation_type='set tags on')
        tags_group = parser.add_mutually_exclusive_group(required=True)
        tags_group.add_argument('--tags', metavar='TAG', type=arg_parsers.ArgList(min_length=1), help='        Specifies strings to be removed from the instance tags.\n        Multiple tags can be removed by repeating this flag.\n        ')
        tags_group.add_argument('--all', action='store_true', default=False, help='Remove all tags from the instance.')

    def CreateReference(self, client, resources, args):
        return flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=flags.GetInstanceZoneScopeLister(client))

    def GetGetRequest(self, client, instance_ref):
        return (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))

    def GetSetRequest(self, client, instance_ref, replacement):
        return (client.apitools_client.instances, 'SetTags', client.messages.ComputeInstancesSetTagsRequest(tags=replacement.tags, **instance_ref.AsDict()))

    def Modify(self, args, existing):
        new_object = encoding.CopyProtoMessage(existing)
        if args.all:
            new_object.tags.items = []
        else:
            new_object.tags.items = sorted(set(new_object.tags.items) - set(args.tags))
        return new_object

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = self.CreateReference(client, holder.resources, args)
        get_request = self.GetGetRequest(client, instance_ref)
        objects = client.MakeRequests([get_request])
        new_object = self.Modify(args, objects[0])
        if not new_object or objects[0] == new_object:
            log.status.Print('No change requested; skipping update for [{0}].'.format(objects[0].name))
            return objects
        return client.MakeRequests([self.GetSetRequest(client, instance_ref, new_object)])
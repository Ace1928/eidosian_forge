from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListSnapshots(base.ListCommand):
    """Lists Cloud Pub/Sub snapshots from a given topic."""
    detailed_help = {'DESCRIPTION': '          Lists all of the Cloud Pub/Sub snapshots attached to the given\n          topic and that match the given filter.', 'EXAMPLES': '          To filter results by snapshot name\n          (ie. only show snapshot \'mysnaps\'), run:\n\n            $ {command} mytopic --filter=snapshotId:mysnaps\n\n          To combine multiple filters (with AND or OR), run:\n\n            $ {command} mytopic --filter="snapshotId:mysnaps1 AND snapshotId:mysnaps2"\n\n          To filter snapshots that match an expression:\n\n            $ {command} mytopic --filter="snapshotId:snaps_*"\n          '}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('yaml')
        parser.display_info.AddUriFunc(util.SnapshotUriFunc)
        resource_args.AddTopicResourceArg(parser, 'to list snapshots for.')

    def Run(self, args):
        client = topics.TopicsClient()
        topic_ref = args.CONCEPTS.topic.Parse()
        return client.ListSnapshots(topic_ref, page_size=args.page_size)
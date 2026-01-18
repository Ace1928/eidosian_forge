from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddStoragePoolCreateArgs(parser, release_track):
    """Add args for creating a Storage Pool."""
    concept_parsers.ConceptParser([flags.GetStoragePoolPresentationSpec('The Storage Pool to create.')]).AddToParser(parser)
    flags.AddResourceDescriptionArg(parser, 'Storage Pool')
    flags.AddResourceCapacityArg(parser, 'Storage Pool')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddCreateLabelsFlags(parser)
    messages = netapp_api_util.GetMessagesModule(release_track=release_track)
    AddStoragePoolServiceLevelArg(parser, messages=messages, required=True)
    AddStoragePoolNetworkArg(parser)
    AddStoragePoolActiveDirectoryArg(parser)
    AddStoragePoolKmsConfigArg(parser)
    AddStoragePoolEnableLdapArg(parser)
    if release_track == base.ReleaseTrack.ALPHA or release_track == base.ReleaseTrack.BETA:
        AddStoragePoolAllowAutoTieringArg(parser)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.images import flags
class DeprecateImages(base.SilentCommand):
    """Manage deprecation status of Compute Engine images.

  *{command}* is used to deprecate images.
  """

    @staticmethod
    def Args(parser):
        DeprecateImages.DISK_IMAGE_ARG = flags.MakeDiskImageArg()
        DeprecateImages.DISK_IMAGE_ARG.AddArgument(parser)
        flags.REPLACEMENT_DISK_IMAGE_ARG.AddArgument(parser)
        deprecation_statuses = {'ACTIVE': 'The image is currently supported.', 'DELETED': 'New uses result in an error. Setting this state will not automatically delete the image. You must still make a request to delete the image to remove it from the image list.', 'DEPRECATED': 'Operations which create a new *DEPRECATED* resource return successfully, but with a warning indicating that the image is deprecated and recommending its replacement.', 'OBSOLETE': 'New uses result in an error.'}
        parser.add_argument('--state', choices=deprecation_statuses, default='ACTIVE', type=lambda x: x.upper(), required=True, help='The deprecation state to set on the image.')
        deprecate_group = parser.add_mutually_exclusive_group()
        deprecate_group.add_argument('--deprecate-on', help="        Specifies a date when the image should be marked as DEPRECATED.\n\n        Note: This is only informational and the image will not be deprecated unless you manually deprecate it.\n\n        This flag is mutually exclusive with *--deprecate-in*.\n\n        The date and time specified must be valid RFC 3339 full-date or date-time.\n        For times in UTC, this looks like ``YYYY-MM-DDTHH:MM:SSZ''.\n        For example: 2020-01-02T00:00:00Z for midnight on January 2, 2020 in UTC.\n        ")
        deprecate_group.add_argument('--deprecate-in', type=arg_parsers.Duration(), help="        Specifies a time duration in which the image should be marked as ``DEPRECATED''.\n\n        Note: This is only informational and the image will not be deprecated unless you manually deprecate it.\n\n        This flag is mutually exclusive with *--deprecate-on*.\n\n        For example, specifying ``30d'' sets the planned ``DEPRECATED'' date to 30 days from the current system time,\n        but does not deprecate the image. You must manually deprecate the image in 30 days.\n        See $ gcloud topic datetimes for information on duration formats.\n\n       ")
        delete_group = parser.add_mutually_exclusive_group()
        delete_group.add_argument('--delete-on', help="        Specifies a date when the image should be marked as ``DELETED''.\n\n        Note: This is only informational and the image will not be deleted unless you manually delete it.\n\n        This flag is mutually exclusive with *--delete-in*.\n\n        The date and time specified must be valid RFC 3339 full-date or date-time.\n        For times in UTC, this looks like ``YYYY-MM-DDTHH:MM:SSZ''.\n        For example: 2020-01-02T00:00:00Z for midnight on January 2, 2020 in UTC.\n\n        ")
        delete_group.add_argument('--delete-in', type=arg_parsers.Duration(), help="        Specifies a time duration in which the image should be marked as ``DELETED''.\n\n        Note: This is only informational and the image will not be deleted unless you manually delete it.\n\n        For example, specifying ``30d'' sets the planned ``DELETED'' time to 30 days from the current system time,\n        but does not delete the image. You must manually delete the image in 30 days.\n        See $ gcloud topic datetimes for information on duration formats.\n\n        This flag is mutually exclusive with *--delete-on*.\n       ")
        obsolete_group = parser.add_mutually_exclusive_group()
        obsolete_group.add_argument('--obsolete-on', help="        Specifies a date when the image should be marked as ``OBSOLETE''.\n\n        Note: This is only informational and the image will not be obsoleted unless you manually obsolete it.\n\n        This flag is mutually exclusive with *--obsolete-in*.\n\n        The date and time specified must be valid RFC 3339 full-date or date-time.\n        For times in UTC, this looks like ``YYYY-MM-DDTHH:MM:SSZ''.\n        For example: 2020-01-02T00:00:00Z for midnight on January 2, 2020 in UTC.\n       ")
        obsolete_group.add_argument('--obsolete-in', type=arg_parsers.Duration(), help="        Specifies a time duration in which the image should be marked as ``OBSOLETE''.\n\n        Note: This is only informational and the image will not be obsoleted unless you manually obsolete it.\n\n        This flag is mutually exclusive with *--obsolete-on*.\n\n        For example, specifying ``30d'' sets the planned ``OBSOLETE'' time to 30 days from the current system time,\n        but does not obsolete the image. You must manually obsolete the image in 30 days.\n        See $ gcloud topic datetimes for information on duration formats.\n        ")

    def Run(self, args):
        """Invokes requests necessary for deprecating images."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        current_time = datetime.datetime.now()
        delete_time = _ResolveTime(args.delete_on, args.delete_in, current_time)
        obsolete_time = _ResolveTime(args.obsolete_on, args.obsolete_in, current_time)
        deprecate_time = _ResolveTime(args.deprecate_on, args.deprecate_in, current_time)
        state = client.messages.DeprecationStatus.StateValueValuesEnum(args.state)
        replacement_ref = flags.REPLACEMENT_DISK_IMAGE_ARG.ResolveAsResource(args, holder.resources)
        if replacement_ref:
            replacement_uri = replacement_ref.SelfLink()
        else:
            replacement_uri = None
        image_ref = DeprecateImages.DISK_IMAGE_ARG.ResolveAsResource(args, holder.resources)
        request = client.messages.ComputeImagesDeprecateRequest(deprecationStatus=client.messages.DeprecationStatus(state=state, deleted=delete_time, obsolete=obsolete_time, deprecated=deprecate_time, replacement=replacement_uri), image=image_ref.Name(), project=image_ref.project)
        return client.MakeRequests([(client.apitools_client.images, 'Deprecate', request)])
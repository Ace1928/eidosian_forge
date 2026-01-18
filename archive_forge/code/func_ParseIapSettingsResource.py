from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def ParseIapSettingsResource(release_track, args):
    """Parse an IAP setting resource from the input arguments.

  Args:
    release_track: base.ReleaseTrack, release track of command.
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Raises:
    calliope_exc.InvalidArgumentException: if `--version` was specified with
        resource type 'backend-services'.

  Returns:
    The specified IapSettingsResource
  """
    if args.organization:
        if args.resource_type:
            raise calliope_exc.InvalidArgumentException('--resource-type', '`--resource-type` should not be specified at organization level')
        if args.project:
            raise calliope_exc.InvalidArgumentException('--project', '`--project` should not be specified at organization level')
        return iap_api.IapSettingsResource(release_track, 'organizations/{0}'.format(args.organization))
    if args.folder:
        if args.resource_type:
            raise calliope_exc.InvalidArgumentException('--resource-type', '`--resource-type` should not be specified at folder level')
        if args.project:
            raise calliope_exc.InvalidArgumentException('--project', '`--project` should not be specified at folder level')
        return iap_api.IapSettingsResource(release_track, 'folders/{0}'.format(args.folder))
    if args.project:
        if not args.resource_type:
            return iap_api.IapSettingsResource(release_track, 'projects/{0}'.format(args.project))
        elif args.resource_type == WEB_RESOURCE_TYPE:
            return iap_api.IapSettingsResource(release_track, 'projects/{0}/iap_web'.format(args.project))
        elif args.resource_type == APP_ENGINE_RESOURCE_TYPE:
            if not args.service:
                return iap_api.IapSettingsResource(release_track, 'projects/{0}/iap_web/appengine-{1}'.format(args.project, args.project))
            elif args.version:
                return iap_api.IapSettingsResource(release_track, 'projects/{0}/iap_web/appengine-{1}/services/{2}/versions/{3}'.format(args.project, args.project, args.service, args.version))
            else:
                return iap_api.IapSettingsResource(release_track, 'projects/{0}/iap_web/appengine-{1}/services/{2}'.format(args.project, args.project, args.service))
        elif args.resource_type == COMPUTE_RESOURCE_TYPE:
            path = ['projects', args.project, 'iap_web']
            if release_track == base.ReleaseTrack.ALPHA and args.region:
                path.append('compute-{}'.format(args.region))
            else:
                path.append('compute')
            if args.service:
                path.extend(['services', args.service])
            return iap_api.IapSettingsResource(release_track, '/'.join(path))
        else:
            raise iap_exc.InvalidIapIamResourceError('Unsupported IAP settings resource type.')
    raise iap_exc.InvalidIapIamResourceError('Could not parse IAP settings resource.')
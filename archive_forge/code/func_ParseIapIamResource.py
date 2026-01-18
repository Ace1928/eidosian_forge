from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def ParseIapIamResource(release_track, args):
    """Parse an IAP IAM resource from the input arguments.

  Args:
    release_track: base.ReleaseTrack, release track of command.
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Raises:
    calliope_exc.InvalidArgumentException: if a provided argument does not apply
        to the specified resource type.
    iap_exc.InvalidIapIamResourceError: if an IapIamResource could not be parsed
        from the arguments.

  Returns:
    The specified IapIamResource
  """
    project = properties.VALUES.core.project.GetOrFail()
    if not args.resource_type:
        if args.service:
            raise calliope_exc.InvalidArgumentException('--service', '`--service` cannot be specified without `--resource-type`.')
        if release_track == base.ReleaseTrack.ALPHA and args.region:
            raise calliope_exc.InvalidArgumentException('--region', '`--region` cannot be specified without `--resource-type`.')
        if args.version:
            raise calliope_exc.InvalidArgumentException('--version', '`--version` cannot be specified without `--resource-type`.')
        return iap_api.IAPWeb(release_track, project)
    elif args.resource_type == APP_ENGINE_RESOURCE_TYPE:
        if release_track == base.ReleaseTrack.ALPHA and args.region:
            raise calliope_exc.InvalidArgumentException('--region', '`--region` cannot be specified for `--resource-type=app-engine`.')
        if args.service and args.version:
            return iap_api.AppEngineServiceVersion(release_track, project, args.service, args.version)
        elif args.service:
            return iap_api.AppEngineService(release_track, project, args.service)
        if args.version:
            raise calliope_exc.InvalidArgumentException('--version', '`--version` cannot be specified without `--service`.')
        return iap_api.AppEngineApplication(release_track, project)
    elif args.resource_type == BACKEND_SERVICES_RESOURCE_TYPE:
        if args.version:
            raise calliope_exc.InvalidArgumentException('--version', '`--version` cannot be specified for `--resource-type=backend-services`.')
        if release_track == base.ReleaseTrack.ALPHA and args.region:
            if args.service:
                return iap_api.BackendService(release_track, project, args.region, args.service)
            else:
                return iap_api.BackendServices(release_track, project, args.region)
        elif args.service:
            return iap_api.BackendService(release_track, project, None, args.service)
        return iap_api.BackendServices(release_track, project, None)
    raise iap_exc.InvalidIapIamResourceError('Could not parse IAP IAM resource.')
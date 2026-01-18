from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetDefaultServiceAccount(base.SilentCommand):
    """Set the default service account on the project.

    *{command}* is used to configure the default service account on project.

  The project's default service account is used when a new instance is
  created unless a custom service account is set via --scopes or
  --no-scopes. Existing instances are not affected.

  For example,

    $ {command} --service-account=example@developers.gserviceaccount.com
    $ gcloud compute instances create instance-name

  will set the project's default service account as
  example@developers.gserviceaccount.com. The instance created will have
  example@developers.gserviceaccount.com as the service account associated
  with because no service account email was specified in the
  "instances create" command.

  To remove the default service account from the project, issue the command:

    $ gcloud compute project-info set-default-service-account \\
        --no-service-account

  The required permission to execute this command is
  `compute.projects.setDefaultServiceAccount`. If needed, you can include this
  permission, or choose any of the following preexisting IAM roles that contain
  this particular permission:

    * Owner
    * Editor
    * Compute Admin
  """

    @staticmethod
    def Args(parser):
        accounts_group = parser.add_mutually_exclusive_group()
        accounts_group.add_argument('--service-account', help='        The email address of the service account that will be set as the default\n        service account for all newly created instances in the project.\n\n        To set the default service account to\n        example@project.iam.gserviceaccount.com:\n\n          $ {command} --service-account example@project.iam.gserviceaccount.com\n        ')
        accounts_group.add_argument('--no-service-account', action='store_true', help='        Sets the default service account on the project as no service account.\n        This causes newly created instances to not run as a service account\n        by default.\n\n        To set the default service account as no service account, specify this\n        flag:\n\n          $ {command} --no-service-account\n        ')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        self.validateFlags(args)
        if args.no_service_account:
            request = client.messages.ComputeProjectsSetDefaultServiceAccountRequest(project=properties.VALUES.core.project.GetOrFail(), projectsSetDefaultServiceAccountRequest=client.messages.ProjectsSetDefaultServiceAccountRequest())
        else:
            request = client.messages.ComputeProjectsSetDefaultServiceAccountRequest(project=properties.VALUES.core.project.GetOrFail(), projectsSetDefaultServiceAccountRequest=client.messages.ProjectsSetDefaultServiceAccountRequest(email=args.service_account))
        return client.MakeRequests([(client.apitools_client.projects, 'SetDefaultServiceAccount', request)])

    def validateFlags(self, args):
        if not args.no_service_account and (not args.service_account):
            raise exceptions.RequiredArgumentException('--service-account', 'must be specified with a service account. To clear the default service account use [--no-service-account].')
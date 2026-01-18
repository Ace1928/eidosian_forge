from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ApplicationDefault(base.Group):
    """Manage your active Application Default Credentials.

  Application Default Credentials (ADC) provide a method to get credentials used
  in calling Google APIs. The {command} command group allows you to manage
  active credentials on your machine that are used for local application
  development.

  These credentials are only used by Google client libraries in your own
  application.

  For more information about ADC and how it works, see [Authenticating as a
  service account](https://cloud.google.com/docs/authentication/production).

  ## EXAMPLES

  To use your own user credentials for your application to access an API, run:

    $ {command} login

  This will take you through a web flow to acquire new user credentials.

  To create a service account and have your application use it for API access,
  run:

    $ gcloud iam service-accounts create my-account
    $ gcloud iam service-accounts keys create key.json
      --iam-account=my-account@my-project.iam.gserviceaccount.com
    $ export GOOGLE_APPLICATION_CREDENTIALS=key.json
    $ ./my_application.sh
  """
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import instances_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
Enable debug mode for an instance (only works on the flexible environment).

  When in debug mode, SSH will be enabled on the VMs, and you can use
  `gcloud compute ssh` to login to them. They will be removed from the health
  checking pools, but they still receive requests.

  Note that any local changes to an instance will be *lost* if debug mode is
  disabled on the instance. New instance(s) may spawn depending on the app's
  scaling settings.

  Additionally, debug mode doesn't work for applications using the
  App Engine standard environment.
  
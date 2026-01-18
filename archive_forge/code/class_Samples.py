from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Samples(base.Group):
    """Cloud Spanner sample apps.

  Each Cloud Spanner sample application includes a backend gRPC service
  backed by a Cloud Spanner database and a workload script that generates
  service traffic.

  These sample apps are open source and available at
  https://github.com/GoogleCloudPlatform/cloud-spanner-samples.

  To see a list of available sample apps, run:

      $ gcloud spanner samples list
  """
    pass
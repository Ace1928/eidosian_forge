from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import store as c_store
class KubeRunAuthException(exceptions.Error):
    """Base Exception for auth issues raised by gcloud kuberun surface."""
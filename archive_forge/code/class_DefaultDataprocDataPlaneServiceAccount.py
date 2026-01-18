from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import iam_helpers
class DefaultDataprocDataPlaneServiceAccount:
    """Find the default Google Service Account used by the Dataproc data plane."""

    @staticmethod
    def Get(project_id):
        return compute_helpers.GetDefaultServiceAccount(project_id)
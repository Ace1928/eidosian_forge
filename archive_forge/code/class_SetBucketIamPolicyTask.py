from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
class SetBucketIamPolicyTask(_SetIamPolicyTask):

    def _make_set_api_call(self, client):
        return client.set_bucket_iam_policy(self._url.bucket_name, self._policy)
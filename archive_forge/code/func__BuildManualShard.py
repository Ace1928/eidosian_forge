from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six
def _BuildManualShard(self, test_targets_for_shard):
    """Build a ManualShard for a ShardingOption."""
    test_targets = [self._BuildTestTargetsForShard(test_target) for test_target in test_targets_for_shard]
    return self._messages.ManualSharding(testTargetsForShard=test_targets)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestTargetsForShard(_messages.Message):
    """Test targets for a shard.

  Fields:
    testTargets: Group of packages, classes, and/or test methods to be run for
      each shard. The targets need to be specified in AndroidJUnitRunner
      argument format. For example, "package com.my.packages" "class
      com.my.package.MyClass". The number of test_targets must be greater than
      0.
  """
    testTargets = _messages.StringField(1, repeated=True)
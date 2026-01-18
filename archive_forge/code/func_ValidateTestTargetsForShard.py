from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def ValidateTestTargetsForShard(args):
    """Validates --test-targets-for-shard uses proper delimiter."""
    if not getattr(args, 'test_targets_for_shard', {}):
        return
    for test_target in args.test_targets_for_shard:
        if _PACKAGE_OR_CLASS_FOLLOWED_BY_COMMA.match(test_target):
            raise test_exceptions.InvalidArgException('test_targets_for_shard', '[{0}] is not a valid test_targets_for_shard argument. Multiple "package" and "class" specifications should be separated by a semicolon instead of a comma.'.format(test_target))
        if _ANY_SPACE_AFTER_COMMA.match(test_target):
            raise test_exceptions.InvalidArgException('test_targets_for_shard', '[{0}] is not a valid test_targets_for_shard argument. No white space is allowed after a comma.'.format(test_target))
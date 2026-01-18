from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
@staticmethod
def ParseVersion(tf_version):
    """Helper to parse the tensorflow version into it's subcomponents."""
    if not tf_version:
        raise TensorflowVersionParser.ParseError('Bad argument: tf_version is empty')
    version_match = TensorflowVersionParser._VERSION_REGEX.match(tf_version)
    nightly_match = TensorflowVersionParser._NIGHTLY_REGEX.match(tf_version)
    if version_match is None and nightly_match is None:
        return TensorflowVersionParser.Result(modifier=tf_version)
    if version_match is not None and nightly_match is not None:
        raise TensorflowVersionParser.ParseError('TF version error: bad version: {}'.format(tf_version))
    if version_match:
        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        result = TensorflowVersionParser.Result(major=major, minor=minor)
        if version_match.group(3):
            patch_match = TensorflowVersionParser._PATCH_NUMBER_REGEX.match(version_match.group(3))
            if patch_match:
                matched_patch = int(patch_match.group(1))
                if matched_patch:
                    result.patch = matched_patch
            else:
                result.modifier = version_match.group(3)
        return result
    if nightly_match:
        result = TensorflowVersionParser.Result(is_nightly=True)
        if nightly_match.group(1):
            result.modifier = nightly_match.group(1)
        return result
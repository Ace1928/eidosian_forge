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
def _ImageFamilyFromTensorflowVersion(self, tf_version, use_dl_image):
    """Generates the image family from the tensorflow version."""
    if tf_version == 'nightly':
        return 'tf-nightly'
    parsed = TensorflowVersionParser.ParseVersion(tf_version)
    if parsed.modifier:
        raise TensorflowVersionParser.ParseError('Invalid tensorflow version:{} (non-empty modifier); please set the --gce-image flag'.format(tf_version))
    if use_dl_image:
        if parsed.major == 2:
            return 'tf2-{}-{}-cpu'.format(parsed.major, parsed.minor)
        else:
            return 'tf-{}-{}-cpu'.format(parsed.major, parsed.minor)
    if parsed.patch or (parsed.major >= 2 and parsed.minor >= 4):
        return 'tf-{}-{}-{}'.format(parsed.major, parsed.minor, parsed.patch)
    return 'tf-{}-{}'.format(parsed.major, parsed.minor)
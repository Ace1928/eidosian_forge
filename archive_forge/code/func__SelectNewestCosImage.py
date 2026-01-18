from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _SelectNewestCosImage(images):
    """Selects newest COS image from the list."""
    cos_images = sorted([image for image in images if image.name.startswith(COS_MAJOR_RELEASE)], key=lambda x: times.ParseDateTime(x.creationTimestamp))
    if not cos_images:
        raise NoCosImageException()
    return cos_images[-1].selfLink
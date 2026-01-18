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
def MakeCreateComparisonKey(name_re):

    def CreateComparisonKey(image):
        version = int(re.match(name_re, image.name).group(1))
        timestamp = times.ParseDateTime(image.creationTimestamp)
        return (version, timestamp)
    return CreateComparisonKey
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddCpu(self):
    self._AddFlag('--cpu', type=arg_parsers.BoundedFloat(lower_bound=0.0), help='Container CPU limit. Limit is expressed as a number of CPUs. Fractional CPU limits are allowed (e.g. 1.5).')
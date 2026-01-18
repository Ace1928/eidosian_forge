from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
Returns whether `field` is handled by this fallthrough class.
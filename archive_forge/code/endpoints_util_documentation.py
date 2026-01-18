from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
import six
Get default output format for prediction results.
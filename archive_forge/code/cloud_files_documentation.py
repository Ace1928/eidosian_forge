from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.code.cloud import cloud
from googlecloudsdk.core import yaml
import six
Generate the Skaffold yaml for the deploy.
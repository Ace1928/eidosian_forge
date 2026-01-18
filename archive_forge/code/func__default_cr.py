from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _default_cr():
    return '\n  apiVersion: operator.run.cloud.google.com/v1alpha1\n  kind: CloudRun\n  metadata:\n    name: cloud-run\n  '
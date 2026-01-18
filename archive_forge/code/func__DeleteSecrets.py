from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
def _DeleteSecrets(secret_map, namespace, context_name):
    kubernetes.DeleteResources('secret', list(secret_map.keys()), namespace, context_name)
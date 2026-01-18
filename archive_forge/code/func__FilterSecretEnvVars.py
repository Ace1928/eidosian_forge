from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
def _FilterSecretEnvVars(env_var):
    return env_var.valueFrom is not None and env_var.valueFrom.secretKeyRef is not None
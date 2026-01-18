from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _ParseAutomaticPolicy(automatic_policy):
    """"Reads automatic replication policy file and returns its data.

  Args:
      automatic_policy (str): The json user managed message

  Returns:
      result (str): "automatic"
      locations (list): empty list
      keys (list): 0 or 1 KMS keys depending on whether the policy has CMEK
  """
    if not automatic_policy:
        return ('automatic', [], [])
    if 'customerManagedEncryption' not in automatic_policy:
        raise exceptions.BadFileException('Failed to parse replication policy. Expected automatic to contain either nothing or customerManagedEncryption.')
    cmek = automatic_policy['customerManagedEncryption']
    if 'kmsKeyName' not in cmek:
        raise exceptions.BadFileException('Failed to find a kmsKeyName in customerManagedEncryption.')
    return ('automatic', [], [cmek['kmsKeyName']])
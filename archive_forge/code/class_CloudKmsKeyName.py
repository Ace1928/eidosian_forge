from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
class CloudKmsKeyName:
    """CloudKmsKeyName to encapsulate `kmsKeyName` and `kmsKeyNames` fields.

  Single `kmsKeyName` and repeated `kmsKeyNames` fields are extracted from user
  input, which are later used in `EncryptionConfig` to pass to Spanner backend.
  """

    def __init__(self, kms_key_name=None, kms_key_names=None):
        self.kms_key_name = kms_key_name
        if kms_key_names is None:
            self.kms_key_names = []
        else:
            self.kms_key_names = kms_key_names
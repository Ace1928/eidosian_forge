from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetAutomatedBackupKmsFlagOverrides():
    return {'kms-key': '--automated-backup-encryption-key', 'kms-keyring': '--automated-backup-encryption-key-keyring', 'kms-location': '--automated-backup-encryption-key-location', 'kms-project': '--automated-backup-encryption-key-project'}
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseUpdatedKmsConfig(self, kms_config, crypto_key_name=None, description=None, labels=None):
    """Parses updates into a new kms config."""
    if crypto_key_name is not None:
        kms_config.cryptoKeyName = crypto_key_name
    if description is not None:
        kms_config.description = description
    if labels is not None:
        kms_config.labels = labels
    return kms_config
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
def GenerateEncryptionConfig(kms_key, dataproc):
    encryption_config = dataproc.messages.GoogleCloudDataprocV1WorkflowTemplateEncryptionConfig()
    encryption_config.kmsKey = kms_key
    return encryption_config
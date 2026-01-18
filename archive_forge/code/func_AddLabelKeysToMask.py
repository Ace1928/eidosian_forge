from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddLabelKeysToMask(labels, request):
    for key in labels:
        request = AddFieldToMask('workerConfig.labels.' + key, request)
    return request
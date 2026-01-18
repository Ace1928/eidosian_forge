from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def RemoveAcceleratorConfig(ref, args, request):
    del ref
    if args.IsSpecified('clear_accelerator_config'):
        request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest.workerPool.workerConfig.accelerator = None
        req = request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest
        AddFieldToMask('workerConfig.accelerator.acceleratorCount', req)
        AddFieldToMask('workerConfig.accelerator.acceleratorType', req)
    return request
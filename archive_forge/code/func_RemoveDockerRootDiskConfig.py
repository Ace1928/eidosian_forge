from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def RemoveDockerRootDiskConfig(ref, args, request):
    del ref
    if args.IsSpecified('clear_docker_root_disk_config'):
        if request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest.workerPool.workerConfig.attachedDisks is not None:
            request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest.workerPool.workerConfig.attachedDisks.dockerRootDisk = None
        req = request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest
        AddFieldToMask('workerConfig.attachedDisks.dockerRootDisk.sourceImage', req)
        AddFieldToMask('workerConfig.attachedDisks.dockerRootDisk.diskType', req)
        AddFieldToMask('workerConfig.attachedDisks.dockerRootDisk.diskSizeGb', req)
    return request
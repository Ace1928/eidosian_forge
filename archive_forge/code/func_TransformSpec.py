from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.run.printers import container_and_volume_printer_util as container_util
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core.resource import custom_printer_base as cp
@staticmethod
def TransformSpec(record):
    limits = container_util.GetLimits(record.template)
    breakglass_value = k8s_util.GetBinAuthzBreakglass(record)
    return cp.Labeled([('Image', record.template.image), ('Tasks', record.spec.taskCount), ('Command', ' '.join(record.template.container.command)), ('Args', ' '.join(record.template.container.args)), ('Binary Authorization', k8s_util.GetBinAuthzPolicy(record)), ('Breakglass Justification', ' ' if breakglass_value == '' else breakglass_value), ('Memory', limits['memory']), ('CPU', limits['cpu']), ('Task Timeout', FormatDurationShort(record.template.spec.timeoutSeconds) if record.template.spec.timeoutSeconds else None), ('Max Retries', '{}'.format(record.template.spec.maxRetries) if record.template.spec.maxRetries is not None else None), ('Parallelism', record.parallelism), ('Service account', record.template.service_account), ('Env vars', container_util.GetUserEnvironmentVariables(record.template)), ('Secrets', container_util.GetSecrets(record.template.container)), ('VPC access', k8s_util.GetVpcNetwork(record.annotations)), ('SQL connections', k8s_util.GetCloudSqlInstances(record.annotations)), ('Volume Mounts', container_util.GetVolumeMounts(record.template.container)), ('Volumes', container_util.GetVolumes(record.template))])
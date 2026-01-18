from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.run.printers import container_and_volume_printer_util as container_util
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core.resource import custom_printer_base as cp
class JobPrinter(cp.CustomPrinterBase):
    """Prints the run Job in a custom human-readable format.

  Format specific to Cloud Run jobs. Only available on Cloud Run commands
  that print jobs.
  """

    @staticmethod
    def TransformSpec(record):
        limits = container_util.GetLimits(record.template)
        breakglass_value = k8s_util.GetBinAuthzBreakglass(record)
        job_spec_annotations = {field.key: field.value for field in record.spec.template.metadata.annotations.additionalProperties}
        return cp.Labeled([('Image', record.template.image), ('Tasks', record.task_count), ('Command', ' '.join(record.template.container.command)), ('Args', ' '.join(record.template.container.args)), ('Binary Authorization', k8s_util.GetBinAuthzPolicy(record)), ('Breakglass Justification', ' ' if breakglass_value == '' else breakglass_value), ('Memory', limits['memory']), ('CPU', limits['cpu']), ('Task Timeout', FormatDurationShort(record.template.spec.timeoutSeconds) if record.template.spec.timeoutSeconds else None), ('Max Retries', '{}'.format(record.max_retries) if record.max_retries is not None else None), ('Parallelism', record.parallelism if record.parallelism else 'No limit'), ('Service account', record.template.service_account), ('Env vars', container_util.GetUserEnvironmentVariables(record.template)), ('Secrets', container_util.GetSecrets(record.template.container)), ('VPC access', k8s_util.GetVpcNetwork(job_spec_annotations)), ('SQL connections', k8s_util.GetCloudSqlInstances(job_spec_annotations)), ('Volume Mounts', container_util.GetVolumeMounts(record.template.container)), ('Volumes', container_util.GetVolumes(record.template))])

    @staticmethod
    def TransformStatus(record):
        if record.status is None:
            return ''
        lines = ['Executed {}'.format(_PluralizedWord('time', record.status.executionCount))]
        if record.status.latestCreatedExecution is not None:
            lines.append('Last executed {} with execution {}'.format(record.status.latestCreatedExecution.creationTimestamp, record.status.latestCreatedExecution.name))
        lines.append(k8s_util.LastUpdatedMessageForJob(record))
        return cp.Lines(lines)

    @staticmethod
    def _formatOutput(record):
        output = []
        header = k8s_util.BuildHeader(record)
        status = JobPrinter.TransformStatus(record)
        labels = k8s_util.GetLabels(record.labels)
        spec = JobPrinter.TransformSpec(record)
        ready_message = k8s_util.FormatReadyMessage(record)
        if header:
            output.append(header)
        if status:
            output.append(status)
        output.append(' ')
        if labels:
            output.append(labels)
            output.append(' ')
        if spec:
            output.append(spec)
        if ready_message:
            output.append(ready_message)
        return output

    def Transform(self, record):
        """Transform a job into the output structure of marker classes."""
        fmt = cp.Lines(JobPrinter._formatOutput(record))
        return fmt
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.command_lib.logs import stream
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_prep
from googlecloudsdk.command_lib.ml_engine import log_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def GetFieldMap(self):
    """Return a mapping of object fields to apitools message fields."""
    return {'masterConfig': {'imageUri': self.master_image_uri, 'acceleratorConfig': {'count': self.master_accelerator_count, 'type': self.master_accelerator_type}}, 'masterType': self.master_machine_type, 'parameterServerConfig': {'imageUri': self.parameter_image_uri, 'acceleratorConfig': {'count': self.parameter_accelerator_count, 'type': self.parameter_accelerator_type}}, 'parameterServerCount': self.parameter_machine_count, 'parameterServerType': self.parameter_machine_type, 'workerConfig': {'imageUri': self.worker_image_uri, 'acceleratorConfig': {'count': self.work_accelerator_count, 'type': self.work_accelerator_type}, 'tpuTfVersion': self.tpu_tf_version}, 'workerCount': self.worker_machine_count, 'workerType': self.worker_machine_type, 'useChiefInTfConfig': self.use_chief_in_tf_config}
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
class TrainingCustomInputServerConfig(object):
    """Data class for passing custom server config for training job input."""

    def __init__(self, runtime_version, scale_tier, master_machine_type=None, master_image_uri=None, master_accelerator_type=None, master_accelerator_count=None, parameter_machine_type=None, parameter_machine_count=None, parameter_image_uri=None, parameter_accelerator_type=None, parameter_accelerator_count=None, tpu_tf_version=None, worker_machine_type=None, worker_machine_count=None, worker_image_uri=None, work_accelerator_type=None, work_accelerator_count=None, use_chief_in_tf_config=None):
        self.master_image_uri = master_image_uri
        self.master_machine_type = master_machine_type
        self.master_accelerator_type = master_accelerator_type
        self.master_accelerator_count = master_accelerator_count
        self.parameter_machine_type = parameter_machine_type
        self.parameter_machine_count = parameter_machine_count
        self.parameter_image_uri = parameter_image_uri
        self.parameter_accelerator_type = parameter_accelerator_type
        self.parameter_accelerator_count = parameter_accelerator_count
        self.tpu_tf_version = tpu_tf_version
        self.worker_machine_type = worker_machine_type
        self.worker_machine_count = worker_machine_count
        self.worker_image_uri = worker_image_uri
        self.work_accelerator_type = work_accelerator_type
        self.work_accelerator_count = work_accelerator_count
        self.runtime_version = runtime_version
        self.scale_tier = scale_tier
        self.use_chief_in_tf_config = use_chief_in_tf_config

    def ValidateConfig(self):
        """Validate that custom config parameters are set correctly."""
        if self.master_image_uri and self.runtime_version:
            raise flags.ArgumentError('Only one of --master-image-uri, --runtime-version can be set.')
        if self.scale_tier and self.scale_tier.name == 'CUSTOM':
            if not self.master_machine_type:
                raise flags.ArgumentError('--master-machine-type is required if scale-tier is set to `CUSTOM`.')
        return True

    def GetFieldMap(self):
        """Return a mapping of object fields to apitools message fields."""
        return {'masterConfig': {'imageUri': self.master_image_uri, 'acceleratorConfig': {'count': self.master_accelerator_count, 'type': self.master_accelerator_type}}, 'masterType': self.master_machine_type, 'parameterServerConfig': {'imageUri': self.parameter_image_uri, 'acceleratorConfig': {'count': self.parameter_accelerator_count, 'type': self.parameter_accelerator_type}}, 'parameterServerCount': self.parameter_machine_count, 'parameterServerType': self.parameter_machine_type, 'workerConfig': {'imageUri': self.worker_image_uri, 'acceleratorConfig': {'count': self.work_accelerator_count, 'type': self.work_accelerator_type}, 'tpuTfVersion': self.tpu_tf_version}, 'workerCount': self.worker_machine_count, 'workerType': self.worker_machine_type, 'useChiefInTfConfig': self.use_chief_in_tf_config}

    @classmethod
    def FromArgs(cls, args, support_tpu_tf_version=False):
        """Build TrainingCustomInputServerConfig from argparse.Namespace."""
        tier = args.scale_tier
        if not tier:
            if args.config:
                data = yaml.load_path(args.config)
                tier = data.get('trainingInput', {}).get('scaleTier', None)
        parsed_tier = ScaleTierFlagMap().GetEnumForChoice(tier)
        return cls(scale_tier=parsed_tier, runtime_version=args.runtime_version, master_machine_type=args.master_machine_type, master_image_uri=args.master_image_uri, master_accelerator_type=args.master_accelerator.get('type') if args.master_accelerator else None, master_accelerator_count=args.master_accelerator.get('count') if args.master_accelerator else None, parameter_machine_type=args.parameter_server_machine_type, parameter_machine_count=args.parameter_server_count, parameter_image_uri=args.parameter_server_image_uri, parameter_accelerator_type=args.parameter_server_accelerator.get('type') if args.parameter_server_accelerator else None, parameter_accelerator_count=args.parameter_server_accelerator.get('count') if args.parameter_server_accelerator else None, tpu_tf_version=args.tpu_tf_version if support_tpu_tf_version else None, worker_machine_type=args.worker_machine_type, worker_machine_count=args.worker_count, worker_image_uri=args.worker_image_uri, work_accelerator_type=args.worker_accelerator.get('type') if args.worker_accelerator else None, work_accelerator_count=args.worker_accelerator.get('count') if args.worker_accelerator else None, use_chief_in_tf_config=args.use_chief_in_tf_config)
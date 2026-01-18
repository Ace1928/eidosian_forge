from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def BuildTrainingJob(self, path=None, module_name=None, job_name=None, trainer_uri=None, region=None, job_dir=None, scale_tier=None, user_args=None, runtime_version=None, python_version=None, network=None, service_account=None, labels=None, kms_key=None, custom_train_server_config=None, enable_web_access=None):
    """Builds a Cloud ML Engine Job from a config file and/or flag values.

    Args:
        path: path to a yaml configuration file
        module_name: value to set for moduleName field (overrides yaml file)
        job_name: value to set for jobName field (overrides yaml file)
        trainer_uri: List of values to set for trainerUri field (overrides yaml
          file)
        region: compute region in which to run the job (overrides yaml file)
        job_dir: Cloud Storage working directory for the job (overrides yaml
          file)
        scale_tier: ScaleTierValueValuesEnum the scale tier for the job
          (overrides yaml file)
        user_args: [str]. A list of arguments to pass through to the job.
        (overrides yaml file)
        runtime_version: the runtime version in which to run the job (overrides
          yaml file)
        python_version: the Python version in which to run the job (overrides
          yaml file)
        network: user network to which the job should be peered with (overrides
          yaml file)
        service_account: A service account (email address string) to use for the
          job.
        labels: Job.LabelsValue, the Cloud labels for the job
        kms_key: A customer-managed encryption key to use for the job.
        custom_train_server_config: jobs_util.CustomTrainingInputServerConfig,
          configuration object for custom server parameters.
        enable_web_access: whether to enable the interactive shell for the job.
    Raises:
      NoPackagesSpecifiedError: if a non-custom job was specified without any
        trainer_uris.
    Returns:
        A constructed Job object.
    """
    job = self.job_class()
    if path:
        data = yaml.load_path(path)
        if data:
            job = encoding.DictToMessage(data, self.job_class)
    if job_name:
        job.jobId = job_name
    if labels is not None:
        job.labels = labels
    if not job.trainingInput:
        job.trainingInput = self.training_input_class()
    additional_fields = {'pythonModule': module_name, 'args': user_args, 'packageUris': trainer_uri, 'region': region, 'jobDir': job_dir, 'scaleTier': scale_tier, 'runtimeVersion': runtime_version, 'pythonVersion': python_version, 'network': network, 'serviceAccount': service_account, 'enableWebAccess': enable_web_access}
    for field_name, value in additional_fields.items():
        if value is not None:
            setattr(job.trainingInput, field_name, value)
    if kms_key:
        arg_utils.SetFieldInMessage(job, 'trainingInput.encryptionConfig.kmsKeyName', kms_key)
    if custom_train_server_config:
        for field_name, value in custom_train_server_config.GetFieldMap().items():
            if value is not None:
                if field_name.endswith('Config') and (not field_name.endswith('TfConfig')):
                    if value['imageUri']:
                        arg_utils.SetFieldInMessage(job, 'trainingInput.{}.imageUri'.format(field_name), value['imageUri'])
                    if value['acceleratorConfig']['type']:
                        arg_utils.SetFieldInMessage(job, 'trainingInput.{}.acceleratorConfig.type'.format(field_name), value['acceleratorConfig']['type'])
                    if value['acceleratorConfig']['count']:
                        arg_utils.SetFieldInMessage(job, 'trainingInput.{}.acceleratorConfig.count'.format(field_name), value['acceleratorConfig']['count'])
                    if field_name == 'workerConfig' and value['tpuTfVersion']:
                        arg_utils.SetFieldInMessage(job, 'trainingInput.{}.tpuTfVersion'.format(field_name), value['tpuTfVersion'])
                else:
                    setattr(job.trainingInput, field_name, value)
    if not self.HasPackageURIs(job) and (not self.IsCustomContainerTraining(job)):
        raise NoPackagesSpecifiedError('Non-custom jobs must have packages.')
    return job
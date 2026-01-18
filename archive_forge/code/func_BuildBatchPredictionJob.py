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
def BuildBatchPredictionJob(self, job_name=None, model_dir=None, model_name=None, version_name=None, input_paths=None, data_format=None, output_path=None, region=None, runtime_version=None, max_worker_count=None, batch_size=None, signature_name=None, labels=None, accelerator_count=None, accelerator_type=None):
    """Builds a Cloud ML Engine Job for batch prediction from flag values.

    Args:
        job_name: value to set for jobName field
        model_dir: str, Google Cloud Storage location of the model files
        model_name: str, value to set for modelName field
        version_name: str, value to set for versionName field
        input_paths: list of input files
        data_format: format of the input files
        output_path: single value for the output location
        region: compute region in which to run the job
        runtime_version: the runtime version in which to run the job
        max_worker_count: int, the maximum number of workers to use
        batch_size: int, the number of records per batch sent to Tensorflow
        signature_name: str, name of input/output signature in the TF meta graph
        labels: Job.LabelsValue, the Cloud labels for the job
        accelerator_count: int, The number of accelerators to attach to the
           machines
       accelerator_type: AcceleratorsValueListEntryValuesEnum, The type of
           accelerator to add to machine.
    Returns:
        A constructed Job object.
    """
    project_id = properties.VALUES.core.project.GetOrFail()
    if accelerator_type:
        accelerator_config_msg = self.GetShortMessage('AcceleratorConfig')
        accelerator_config = accelerator_config_msg(count=accelerator_count, type=accelerator_type)
    else:
        accelerator_config = None
    prediction_input = self.prediction_input_class(inputPaths=input_paths, outputPath=output_path, region=region, runtimeVersion=runtime_version, maxWorkerCount=max_worker_count, batchSize=batch_size, accelerator=accelerator_config)
    prediction_input.dataFormat = prediction_input.DataFormatValueValuesEnum(data_format)
    if model_dir:
        prediction_input.uri = model_dir
    elif version_name:
        version_ref = resources.REGISTRY.Parse(version_name, collection='ml.projects.models.versions', params={'modelsId': model_name, 'projectsId': project_id})
        prediction_input.versionName = version_ref.RelativeName()
    else:
        model_ref = resources.REGISTRY.Parse(model_name, collection='ml.projects.models', params={'projectsId': project_id})
        prediction_input.modelName = model_ref.RelativeName()
    if signature_name:
        prediction_input.signatureName = signature_name
    return self.job_class(jobId=job_name, predictionInput=prediction_input, labels=labels)
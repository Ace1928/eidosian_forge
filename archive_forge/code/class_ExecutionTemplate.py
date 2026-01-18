from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionTemplate(_messages.Message):
    """The description a notebook execution workload.

  Enums:
    JobTypeValueValuesEnum: The type of Job to be used on this execution.
    ScaleTierValueValuesEnum: Required. Scale tier of the hardware used for
      notebook execution. DEPRECATED Will be discontinued. As right now only
      CUSTOM is supported.

  Messages:
    LabelsValue: Labels for execution. If execution is scheduled, a field
      included will be 'nbs-scheduled'. Otherwise, it is an immediate
      execution, and an included field will be 'nbs-immediate'. Use fields to
      efficiently index between various types of executions.

  Fields:
    acceleratorConfig: Configuration (count and accelerator type) for hardware
      running notebook execution.
    containerImageUri: Container Image URI to a DLVM Example:
      'gcr.io/deeplearning-platform-release/base-cu100' More examples can be
      found at: https://cloud.google.com/ai-platform/deep-learning-
      containers/docs/choosing-container
    dataprocParameters: Parameters used in Dataproc JobType executions.
    inputNotebookFile: Path to the notebook file to execute. Must be in a
      Google Cloud Storage bucket. Format:
      `gs://{bucket_name}/{folder}/{notebook_file_name}` Ex:
      `gs://notebook_user/scheduled_notebooks/sentiment_notebook.ipynb`
    jobType: The type of Job to be used on this execution.
    kernelSpec: Name of the kernel spec to use. This must be specified if the
      kernel spec name on the execution target does not match the name in the
      input notebook file.
    labels: Labels for execution. If execution is scheduled, a field included
      will be 'nbs-scheduled'. Otherwise, it is an immediate execution, and an
      included field will be 'nbs-immediate'. Use fields to efficiently index
      between various types of executions.
    masterType: Specifies the type of virtual machine to use for your training
      job's master worker. You must specify this field when `scaleTier` is set
      to `CUSTOM`. You can use certain Compute Engine machine types directly
      in this field. The following types are supported: - `n1-standard-4` -
      `n1-standard-8` - `n1-standard-16` - `n1-standard-32` - `n1-standard-64`
      - `n1-standard-96` - `n1-highmem-2` - `n1-highmem-4` - `n1-highmem-8` -
      `n1-highmem-16` - `n1-highmem-32` - `n1-highmem-64` - `n1-highmem-96` -
      `n1-highcpu-16` - `n1-highcpu-32` - `n1-highcpu-64` - `n1-highcpu-96`
      Alternatively, you can use the following legacy machine types: -
      `standard` - `large_model` - `complex_model_s` - `complex_model_m` -
      `complex_model_l` - `standard_gpu` - `complex_model_m_gpu` -
      `complex_model_l_gpu` - `standard_p100` - `complex_model_m_p100` -
      `standard_v100` - `large_model_v100` - `complex_model_m_v100` -
      `complex_model_l_v100` Finally, if you want to use a TPU for training,
      specify `cloud_tpu` in this field. Learn more about the [special
      configuration options for training with
      TPU](https://cloud.google.com/ai-platform/training/docs/using-
      tpus#configuring_a_custom_tpu_machine).
    outputNotebookFolder: Path to the notebook folder to write to. Must be in
      a Google Cloud Storage bucket path. Format:
      `gs://{bucket_name}/{folder}` Ex:
      `gs://notebook_user/scheduled_notebooks`
    parameters: Parameters used within the 'input_notebook_file' notebook.
    paramsYamlFile: Parameters to be overridden in the notebook during
      execution. Ref https://papermill.readthedocs.io/en/latest/usage-
      parameterize.html on how to specifying parameters in the input notebook
      and pass them here in an YAML file. Ex:
      `gs://notebook_user/scheduled_notebooks/sentiment_notebook_params.yaml`
    scaleTier: Required. Scale tier of the hardware used for notebook
      execution. DEPRECATED Will be discontinued. As right now only CUSTOM is
      supported.
    serviceAccount: The email address of a service account to use when running
      the execution. You must have the `iam.serviceAccounts.actAs` permission
      for the specified service account.
    tensorboard: The name of a Vertex AI [Tensorboard] resource to which this
      execution will upload Tensorboard logs. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
    vertexAiParameters: Parameters used in Vertex AI JobType executions.
  """

    class JobTypeValueValuesEnum(_messages.Enum):
        """The type of Job to be used on this execution.

    Values:
      JOB_TYPE_UNSPECIFIED: No type specified.
      VERTEX_AI: Custom Job in `aiplatform.googleapis.com`. Default value for
        an execution.
      DATAPROC: Run execution on a cluster with Dataproc as a job. https://clo
        ud.google.com/dataproc/docs/reference/rest/v1/projects.regions.jobs
    """
        JOB_TYPE_UNSPECIFIED = 0
        VERTEX_AI = 1
        DATAPROC = 2

    class ScaleTierValueValuesEnum(_messages.Enum):
        """Required. Scale tier of the hardware used for notebook execution.
    DEPRECATED Will be discontinued. As right now only CUSTOM is supported.

    Values:
      SCALE_TIER_UNSPECIFIED: Unspecified Scale Tier.
      BASIC: A single worker instance. This tier is suitable for learning how
        to use Cloud ML, and for experimenting with new models using small
        datasets.
      STANDARD_1: Many workers and a few parameter servers.
      PREMIUM_1: A large number of workers with many parameter servers.
      BASIC_GPU: A single worker instance with a K80 GPU.
      BASIC_TPU: A single worker instance with a Cloud TPU.
      CUSTOM: The CUSTOM tier is not a set tier, but rather enables you to use
        your own cluster specification. When you use this tier, set values to
        configure your processing cluster according to these guidelines: * You
        _must_ set `ExecutionTemplate.masterType` to specify the type of
        machine to use for your master node. This is the only required
        setting.
    """
        SCALE_TIER_UNSPECIFIED = 0
        BASIC = 1
        STANDARD_1 = 2
        PREMIUM_1 = 3
        BASIC_GPU = 4
        BASIC_TPU = 5
        CUSTOM = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for execution. If execution is scheduled, a field included will
    be 'nbs-scheduled'. Otherwise, it is an immediate execution, and an
    included field will be 'nbs-immediate'. Use fields to efficiently index
    between various types of executions.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    acceleratorConfig = _messages.MessageField('SchedulerAcceleratorConfig', 1)
    containerImageUri = _messages.StringField(2)
    dataprocParameters = _messages.MessageField('DataprocParameters', 3)
    inputNotebookFile = _messages.StringField(4)
    jobType = _messages.EnumField('JobTypeValueValuesEnum', 5)
    kernelSpec = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    masterType = _messages.StringField(8)
    outputNotebookFolder = _messages.StringField(9)
    parameters = _messages.StringField(10)
    paramsYamlFile = _messages.StringField(11)
    scaleTier = _messages.EnumField('ScaleTierValueValuesEnum', 12)
    serviceAccount = _messages.StringField(13)
    tensorboard = _messages.StringField(14)
    vertexAiParameters = _messages.MessageField('VertexAIParameters', 15)
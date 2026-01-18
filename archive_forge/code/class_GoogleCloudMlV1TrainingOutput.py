from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1TrainingOutput(_messages.Message):
    """Represents results of a training job. Output only.

  Messages:
    WebAccessUrisValue: Output only. URIs for accessing [interactive
      shells](https://cloud.google.com/ai-platform/training/docs/monitor-
      debug-interactive-shell) (one URI for each training node). Only
      available if training_input.enable_web_access is `true`. The keys are
      names of each node in the training job; for example, `master-replica-0`
      for the master node, `worker-replica-0` for the first worker, and `ps-
      replica-0` for the first parameter server. The values are the URIs for
      each node's interactive shell.

  Fields:
    builtInAlgorithmOutput: Details related to built-in algorithms jobs. Only
      set for built-in algorithms jobs.
    completedTrialCount: The number of hyperparameter tuning trials that
      completed successfully. Only set for hyperparameter tuning jobs.
    consumedMLUnits: The amount of ML units consumed by the job.
    hyperparameterMetricTag: The TensorFlow summary tag name used for
      optimizing hyperparameter tuning trials. See [`HyperparameterSpec.hyperp
      arameterMetricTag`](#HyperparameterSpec.FIELDS.hyperparameter_metric_tag
      ) for more information. Only set for hyperparameter tuning jobs.
    isBuiltInAlgorithmJob: Whether this job is a built-in Algorithm job.
    isHyperparameterTuningJob: Whether this job is a hyperparameter tuning
      job.
    nasJobOutput: The output of a Neural Architecture Search (NAS) job.
    trials: Results for individual Hyperparameter trials. Only set for
      hyperparameter tuning jobs.
    webAccessUris: Output only. URIs for accessing [interactive
      shells](https://cloud.google.com/ai-platform/training/docs/monitor-
      debug-interactive-shell) (one URI for each training node). Only
      available if training_input.enable_web_access is `true`. The keys are
      names of each node in the training job; for example, `master-replica-0`
      for the master node, `worker-replica-0` for the first worker, and `ps-
      replica-0` for the first parameter server. The values are the URIs for
      each node's interactive shell.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class WebAccessUrisValue(_messages.Message):
        """Output only. URIs for accessing [interactive
    shells](https://cloud.google.com/ai-platform/training/docs/monitor-debug-
    interactive-shell) (one URI for each training node). Only available if
    training_input.enable_web_access is `true`. The keys are names of each
    node in the training job; for example, `master-replica-0` for the master
    node, `worker-replica-0` for the first worker, and `ps-replica-0` for the
    first parameter server. The values are the URIs for each node's
    interactive shell.

    Messages:
      AdditionalProperty: An additional property for a WebAccessUrisValue
        object.

    Fields:
      additionalProperties: Additional properties of type WebAccessUrisValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a WebAccessUrisValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    builtInAlgorithmOutput = _messages.MessageField('GoogleCloudMlV1BuiltInAlgorithmOutput', 1)
    completedTrialCount = _messages.IntegerField(2)
    consumedMLUnits = _messages.FloatField(3)
    hyperparameterMetricTag = _messages.StringField(4)
    isBuiltInAlgorithmJob = _messages.BooleanField(5)
    isHyperparameterTuningJob = _messages.BooleanField(6)
    nasJobOutput = _messages.MessageField('GoogleCloudMlV1NasJobOutput', 7)
    trials = _messages.MessageField('GoogleCloudMlV1HyperparameterOutput', 8, repeated=True)
    webAccessUris = _messages.MessageField('WebAccessUrisValue', 9)
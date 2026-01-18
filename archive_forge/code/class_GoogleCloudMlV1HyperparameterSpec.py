from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1HyperparameterSpec(_messages.Message):
    """Represents a set of hyperparameters to optimize.

  Enums:
    AlgorithmValueValuesEnum: Optional. The search algorithm specified for the
      hyperparameter tuning job. Uses the default AI Platform hyperparameter
      tuning algorithm if unspecified.
    GoalValueValuesEnum: Required. The type of goal to use for tuning.
      Available types are `MAXIMIZE` and `MINIMIZE`. Defaults to `MAXIMIZE`.

  Fields:
    algorithm: Optional. The search algorithm specified for the hyperparameter
      tuning job. Uses the default AI Platform hyperparameter tuning algorithm
      if unspecified.
    enableTrialEarlyStopping: Optional. Indicates if the hyperparameter tuning
      job enables auto trial early stopping.
    goal: Required. The type of goal to use for tuning. Available types are
      `MAXIMIZE` and `MINIMIZE`. Defaults to `MAXIMIZE`.
    hyperparameterMetricTag: Optional. The TensorFlow summary tag name to use
      for optimizing trials. For current versions of TensorFlow, this tag name
      should exactly match what is shown in TensorBoard, including all scopes.
      For versions of TensorFlow prior to 0.12, this should be only the tag
      passed to tf.Summary. By default, "training/hptuning/metric" will be
      used.
    maxFailedTrials: Optional. The number of failed trials that need to be
      seen before failing the hyperparameter tuning job. You can specify this
      field to override the default failing criteria for AI Platform
      hyperparameter tuning jobs. Defaults to zero, which means the service
      decides when a hyperparameter job should fail.
    maxParallelTrials: Optional. The number of training trials to run
      concurrently. You can reduce the time it takes to perform hyperparameter
      tuning by adding trials in parallel. However, each trail only benefits
      from the information gained in completed trials. That means that a trial
      does not get access to the results of trials running at the same time,
      which could reduce the quality of the overall optimization. Each trial
      will use the same scale tier and machine types. Defaults to one.
    maxTrials: Optional. How many training trials should be attempted to
      optimize the specified hyperparameters. Defaults to one.
    params: Required. The set of parameters to tune.
    resumePreviousJobId: Optional. The prior hyperparameter tuning job id that
      users hope to continue with. The job id will be used to find the
      corresponding vizier study guid and resume the study.
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """Optional. The search algorithm specified for the hyperparameter tuning
    job. Uses the default AI Platform hyperparameter tuning algorithm if
    unspecified.

    Values:
      ALGORITHM_UNSPECIFIED: The default algorithm used by the hyperparameter
        tuning service. This is a Bayesian optimization algorithm.
      GRID_SEARCH: Simple grid search within the feasible space. To use grid
        search, all parameters must be `INTEGER`, `CATEGORICAL`, or
        `DISCRETE`.
      RANDOM_SEARCH: Simple random search within the feasible space.
      POPULATION_BASED_TRAINING: Population Based Training Algorithm.
    """
        ALGORITHM_UNSPECIFIED = 0
        GRID_SEARCH = 1
        RANDOM_SEARCH = 2
        POPULATION_BASED_TRAINING = 3

    class GoalValueValuesEnum(_messages.Enum):
        """Required. The type of goal to use for tuning. Available types are
    `MAXIMIZE` and `MINIMIZE`. Defaults to `MAXIMIZE`.

    Values:
      GOAL_TYPE_UNSPECIFIED: Goal Type will default to maximize.
      MAXIMIZE: Maximize the goal metric.
      MINIMIZE: Minimize the goal metric.
    """
        GOAL_TYPE_UNSPECIFIED = 0
        MAXIMIZE = 1
        MINIMIZE = 2
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    enableTrialEarlyStopping = _messages.BooleanField(2)
    goal = _messages.EnumField('GoalValueValuesEnum', 3)
    hyperparameterMetricTag = _messages.StringField(4)
    maxFailedTrials = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    maxParallelTrials = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    maxTrials = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    params = _messages.MessageField('GoogleCloudMlV1ParameterSpec', 8, repeated=True)
    resumePreviousJobId = _messages.StringField(9)
from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlStatistics(_messages.Message):
    """Job statistics specific to a BigQuery ML training job.

  Enums:
    ModelTypeValueValuesEnum: Output only. The type of the model that is being
      trained.
    TrainingTypeValueValuesEnum: Output only. Training type of the job.

  Fields:
    hparamTrials: Output only. Trials of a [hyperparameter tuning
      job](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-
      tuning-overview) sorted by trial_id.
    iterationResults: Results for all completed iterations. Empty for
      [hyperparameter tuning jobs](/bigquery-ml/docs/reference/standard-
      sql/bigqueryml-syntax-hp-tuning-overview).
    maxIterations: Output only. Maximum number of iterations specified as
      max_iterations in the 'CREATE MODEL' query. The actual number of
      iterations may be less than this number due to early stop.
    modelType: Output only. The type of the model that is being trained.
    trainingType: Output only. Training type of the job.
  """

    class ModelTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the model that is being trained.

    Values:
      MODEL_TYPE_UNSPECIFIED: Default value.
      LINEAR_REGRESSION: Linear regression model.
      LOGISTIC_REGRESSION: Logistic regression based classification model.
      KMEANS: K-means clustering model.
      MATRIX_FACTORIZATION: Matrix factorization model.
      DNN_CLASSIFIER: DNN classifier model.
      TENSORFLOW: An imported TensorFlow model.
      DNN_REGRESSOR: DNN regressor model.
      XGBOOST: An imported XGBoost model.
      BOOSTED_TREE_REGRESSOR: Boosted tree regressor model.
      BOOSTED_TREE_CLASSIFIER: Boosted tree classifier model.
      ARIMA: ARIMA model.
      AUTOML_REGRESSOR: AutoML Tables regression model.
      AUTOML_CLASSIFIER: AutoML Tables classification model.
      PCA: Prinpical Component Analysis model.
      DNN_LINEAR_COMBINED_CLASSIFIER: Wide-and-deep classifier model.
      DNN_LINEAR_COMBINED_REGRESSOR: Wide-and-deep regressor model.
      AUTOENCODER: Autoencoder model.
      ARIMA_PLUS: New name for the ARIMA model.
      ARIMA_PLUS_XREG: ARIMA with external regressors.
      RANDOM_FOREST_REGRESSOR: Random forest regressor model.
      RANDOM_FOREST_CLASSIFIER: Random forest classifier model.
      TENSORFLOW_LITE: An imported TensorFlow Lite model.
      ONNX: An imported ONNX model.
    """
        MODEL_TYPE_UNSPECIFIED = 0
        LINEAR_REGRESSION = 1
        LOGISTIC_REGRESSION = 2
        KMEANS = 3
        MATRIX_FACTORIZATION = 4
        DNN_CLASSIFIER = 5
        TENSORFLOW = 6
        DNN_REGRESSOR = 7
        XGBOOST = 8
        BOOSTED_TREE_REGRESSOR = 9
        BOOSTED_TREE_CLASSIFIER = 10
        ARIMA = 11
        AUTOML_REGRESSOR = 12
        AUTOML_CLASSIFIER = 13
        PCA = 14
        DNN_LINEAR_COMBINED_CLASSIFIER = 15
        DNN_LINEAR_COMBINED_REGRESSOR = 16
        AUTOENCODER = 17
        ARIMA_PLUS = 18
        ARIMA_PLUS_XREG = 19
        RANDOM_FOREST_REGRESSOR = 20
        RANDOM_FOREST_CLASSIFIER = 21
        TENSORFLOW_LITE = 22
        ONNX = 23

    class TrainingTypeValueValuesEnum(_messages.Enum):
        """Output only. Training type of the job.

    Values:
      TRAINING_TYPE_UNSPECIFIED: Unspecified training type.
      SINGLE_TRAINING: Single training with fixed parameter space.
      HPARAM_TUNING: [Hyperparameter tuning training](/bigquery-
        ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview).
    """
        TRAINING_TYPE_UNSPECIFIED = 0
        SINGLE_TRAINING = 1
        HPARAM_TUNING = 2
    hparamTrials = _messages.MessageField('HparamTuningTrial', 1, repeated=True)
    iterationResults = _messages.MessageField('IterationResult', 2, repeated=True)
    maxIterations = _messages.IntegerField(3)
    modelType = _messages.EnumField('ModelTypeValueValuesEnum', 4)
    trainingType = _messages.EnumField('TrainingTypeValueValuesEnum', 5)
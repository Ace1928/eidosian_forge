from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputs(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionSeq2SeqPlusFore
  castingInputs object.

  Fields:
    additionalExperiments: Additional experiment flags for the time series
      forcasting training.
    availableAtForecastColumns: Names of columns that are available and
      provided when a forecast is requested. These columns contain information
      for the given entity (identified by the time_series_identifier_column
      column) that is known at forecast. For example, predicted weather for a
      specific day.
    contextWindow: The amount of time into the past training and prediction
      data is used for model training and prediction respectively. Expressed
      in number of units defined by the `data_granularity` field.
    dataGranularity: Expected difference in time granularity between rows in
      the data.
    exportEvaluatedDataItemsConfig: Configuration for exporting test set
      predictions to a BigQuery table. If this configuration is absent, then
      the export is not performed.
    forecastHorizon: The amount of time into the future for which forecasted
      values for the target are returned. Expressed in number of units defined
      by the `data_granularity` field.
    hierarchyConfig: Configuration that defines the hierarchical relationship
      of time series and parameters for hierarchical forecasting strategies.
    holidayRegions: The geographical region based on which the holiday effect
      is applied in modeling by adding holiday categorical array feature that
      include all holidays matching the date. This option only allowed when
      data_granularity is day. By default, holiday effect modeling is
      disabled. To turn it on, specify the holiday region using this option.
    optimizationObjective: Objective function the model is optimizing towards.
      The training process creates a model that optimizes the value of the
      objective function over the validation set. The supported optimization
      objectives: * "minimize-rmse" (default) - Minimize root-mean-squared
      error (RMSE). * "minimize-mae" - Minimize mean-absolute error (MAE). *
      "minimize-rmsle" - Minimize root-mean-squared log error (RMSLE). *
      "minimize-rmspe" - Minimize root-mean-squared percentage error (RMSPE).
      * "minimize-wape-mae" - Minimize the combination of weighted absolute
      percentage error (WAPE) and mean-absolute-error (MAE). * "minimize-
      quantile-loss" - Minimize the quantile loss at the quantiles defined in
      `quantiles`. * "minimize-mape" - Minimize the mean absolute percentage
      error.
    quantiles: Quantiles to use for minimize-quantile-loss
      `optimization_objective`. Up to 5 quantiles are allowed of values
      between 0 and 1, exclusive. Required if the value of
      optimization_objective is minimize-quantile-loss. Represents the percent
      quantiles to use for that objective. Quantiles must be unique.
    targetColumn: The name of the column that the Model is to predict values
      for. This column must be unavailable at forecast.
    timeColumn: The name of the column that identifies time order in the time
      series. This column must be available at forecast.
    timeSeriesAttributeColumns: Column names that should be used as attribute
      columns. The value of these columns does not vary as a function of time.
      For example, store ID or item color.
    timeSeriesIdentifierColumn: The name of the column that identifies the
      time series.
    trainBudgetMilliNodeHours: Required. The train budget of creating this
      model, expressed in milli node hours i.e. 1,000 value in this field
      means 1 node hour. The training cost of the model will not exceed this
      budget. The final cost will be attempted to be close to the budget,
      though may end up being (even) noticeably smaller - at the backend's
      discretion. This especially may happen when further model training
      ceases to provide any improvements. If the budget is set to a value
      known to be insufficient to train a model for the given dataset, the
      training won't be attempted and will error. The train budget must be
      between 1,000 and 72,000 milli node hours, inclusive.
    transformations: Each transformation will apply transform function to
      given input column. And the result will be used for training. When
      creating transformation for BigQuery Struct column, the column should be
      flattened using "." as the delimiter.
    unavailableAtForecastColumns: Names of columns that are unavailable when a
      forecast is requested. This column contains information for the given
      entity (identified by the time_series_identifier_column) that is unknown
      before the forecast For example, actual weather on a given day.
    validationOptions: Validation options for the data validation component.
      The available options are: * "fail-pipeline" - default, will validate
      against the validation and fail the pipeline if it fails. * "ignore-
      validation" - ignore the results of the validation and continue
    weightColumn: Column name that should be used as the weight column. Higher
      values in this column give more importance to the row during model
      training. The column must have numeric values between 0 and 10000
      inclusively; 0 means the row is ignored for training. If weight column
      field is not set, then all rows are assumed to have equal weight of 1.
      This column must be available at forecast.
    windowConfig: Config containing strategy for generating sliding windows.
  """
    additionalExperiments = _messages.StringField(1, repeated=True)
    availableAtForecastColumns = _messages.StringField(2, repeated=True)
    contextWindow = _messages.IntegerField(3)
    dataGranularity = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsGranularity', 4)
    exportEvaluatedDataItemsConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionExportEvaluatedDataItemsConfig', 5)
    forecastHorizon = _messages.IntegerField(6)
    hierarchyConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionHierarchyConfig', 7)
    holidayRegions = _messages.StringField(8, repeated=True)
    optimizationObjective = _messages.StringField(9)
    quantiles = _messages.FloatField(10, repeated=True)
    targetColumn = _messages.StringField(11)
    timeColumn = _messages.StringField(12)
    timeSeriesAttributeColumns = _messages.StringField(13, repeated=True)
    timeSeriesIdentifierColumn = _messages.StringField(14)
    trainBudgetMilliNodeHours = _messages.IntegerField(15)
    transformations = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformation', 16, repeated=True)
    unavailableAtForecastColumns = _messages.StringField(17, repeated=True)
    validationOptions = _messages.StringField(18)
    weightColumn = _messages.StringField(19)
    windowConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionWindowConfig', 20)
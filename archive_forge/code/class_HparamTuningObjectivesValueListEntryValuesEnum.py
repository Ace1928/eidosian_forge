from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HparamTuningObjectivesValueListEntryValuesEnum(_messages.Enum):
    """HparamTuningObjectivesValueListEntryValuesEnum enum type.

    Values:
      HPARAM_TUNING_OBJECTIVE_UNSPECIFIED: Unspecified evaluation metric.
      MEAN_ABSOLUTE_ERROR: Mean absolute error. mean_absolute_error =
        AVG(ABS(label - predicted))
      MEAN_SQUARED_ERROR: Mean squared error. mean_squared_error =
        AVG(POW(label - predicted, 2))
      MEAN_SQUARED_LOG_ERROR: Mean squared log error. mean_squared_log_error =
        AVG(POW(LN(1 + label) - LN(1 + predicted), 2))
      MEDIAN_ABSOLUTE_ERROR: Mean absolute error. median_absolute_error =
        APPROX_QUANTILES(absolute_error, 2)[OFFSET(1)]
      R_SQUARED: R^2 score. This corresponds to r2_score in ML.EVALUATE.
        r_squared = 1 - SUM(squared_error)/(COUNT(label)*VAR_POP(label))
      EXPLAINED_VARIANCE: Explained variance. explained_variance = 1 -
        VAR_POP(label_error)/VAR_POP(label)
      PRECISION: Precision is the fraction of actual positive predictions that
        had positive actual labels. For multiclass this is a macro-averaged
        metric treating each class as a binary classifier.
      RECALL: Recall is the fraction of actual positive labels that were given
        a positive prediction. For multiclass this is a macro-averaged metric.
      ACCURACY: Accuracy is the fraction of predictions given the correct
        label. For multiclass this is a globally micro-averaged metric.
      F1_SCORE: The F1 score is an average of recall and precision. For
        multiclass this is a macro-averaged metric.
      LOG_LOSS: Logorithmic Loss. For multiclass this is a macro-averaged
        metric.
      ROC_AUC: Area Under an ROC Curve. For multiclass this is a macro-
        averaged metric.
      DAVIES_BOULDIN_INDEX: Davies-Bouldin Index.
      MEAN_AVERAGE_PRECISION: Mean Average Precision.
      NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN: Normalized Discounted Cumulative
        Gain.
      AVERAGE_RANK: Average Rank.
    """
    HPARAM_TUNING_OBJECTIVE_UNSPECIFIED = 0
    MEAN_ABSOLUTE_ERROR = 1
    MEAN_SQUARED_ERROR = 2
    MEAN_SQUARED_LOG_ERROR = 3
    MEDIAN_ABSOLUTE_ERROR = 4
    R_SQUARED = 5
    EXPLAINED_VARIANCE = 6
    PRECISION = 7
    RECALL = 8
    ACCURACY = 9
    F1_SCORE = 10
    LOG_LOSS = 11
    ROC_AUC = 12
    DAVIES_BOULDIN_INDEX = 13
    MEAN_AVERAGE_PRECISION = 14
    NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 15
    AVERAGE_RANK = 16
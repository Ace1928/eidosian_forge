import abc
import types
import warnings
import numpy as np
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.saving.saved_model import metric_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class AUC(Metric):
    """Approximates the AUC (Area under the curve) of the ROC or PR curves.

  The AUC (Area under the curve) of the ROC (Receiver operating
  characteristic; default) or PR (Precision Recall) curves are quality measures
  of binary classifiers. Unlike the accuracy, and like cross-entropy
  losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

  This class approximates AUCs using a Riemann sum. During the metric
  accumulation phrase, predictions are accumulated within predefined buckets
  by value. The AUC is then computed by interpolating per-bucket averages. These
  buckets define the evaluated operational points.

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the AUC.
  To discretize the AUC curve, a linearly spaced set of thresholds is used to
  compute pairs of recall and precision values. The area under the ROC-curve is
  therefore computed using the height of the recall values by the false positive
  rate, while the area under the PR-curve is the computed using the height of
  the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`. The `thresholds` parameter can be
  used to manually specify thresholds which split the predictions more evenly.

  For a best approximation of the real AUC, `predictions` should be distributed
  approximately uniformly in the range [0, 1] (if `from_logits=False`). The
  quality of the AUC approximation may be poor if this is not the case. Setting
  `summation_method` to 'minoring' or 'majoring' can help quantify the error in
  the approximation by providing lower or upper bound estimate of the AUC.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to
      use when discretizing the roc curve. Values must be > 1.
    curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
      [default] or 'PR' for the Precision-Recall-curve.
    summation_method: (Optional) Specifies the [Riemann summation method](
        https://en.wikipedia.org/wiki/Riemann_sum) used.
        'interpolation' (default) applies mid-point summation scheme for `ROC`.
        For PR-AUC, interpolates (true/false) positives but not the ratio that
        is precision (see Davis & Goadrich 2006 for details);
        'minoring' applies left summation
        for increasing intervals and right summation for decreasing intervals;
        'majoring' does the opposite.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    thresholds: (Optional) A list of floating point values to use as the
      thresholds for discretizing the curve. If set, the `num_thresholds`
      parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
      equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
      be automatically included with these to correctly handle predictions
      equal to exactly 0 or 1.
    multi_label: boolean indicating whether multilabel data should be
      treated as such, wherein AUC is computed separately for each label and
      then averaged across labels, or (when False) if the data should be
      flattened into a single label before AUC computation. In the latter
      case, when multilabel data is passed to AUC, each label-prediction pair
      is treated as an individual data point. Should be set to False for
      multi-class data.
    num_labels: (Optional) The number of labels, used when `multi_label` is
      True. If `num_labels` is not specified, then state variables get created
      on the first call to `update_state`.
    label_weights: (Optional) list, array, or tensor of non-negative weights
      used to compute AUCs for multilabel data. When `multi_label` is True,
      the weights are applied to the individual label AUCs when they are
      averaged to produce the multi-label AUC. When it's False, they are used
      to weight the individual label predictions in computing the confusion
      matrix on the flattened data. Note that this is unlike class_weights in
      that class_weights weights the example depending on the value of its
      label, whereas label_weights depends only on the index of that label
      before flattening; therefore `label_weights` should not be used for
      multi-class data.
    from_logits: boolean indicating whether the predictions (`y_pred` in
      `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
      when using a keras loss, the `from_logits` constructor argument of the
      loss should match the AUC `from_logits` constructor argument.

  Standalone usage:

  >>> m = tf.keras.metrics.AUC(num_thresholds=3)
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
  >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
  >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
  >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
  >>> # auc = ((((1+0.5)/2)*(1-0)) + (((0.5+0)/2)*(0-0))) = 0.75
  >>> m.result().numpy()
  0.75

  >>> m.reset_state()
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
  ...                sample_weight=[1, 0, 0, 1])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  # Reports the AUC of a model outputing a probability.
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()])

  # Reports the AUC of a model outputing a logit.
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.AUC(from_logits=True)])
  ```
  """

    def __init__(self, num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, thresholds=None, multi_label=False, num_labels=None, label_weights=None, from_logits=False):
        if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(metrics_utils.AUCCurve):
            raise ValueError('Invalid curve: "{}". Valid options are: "{}"'.format(curve, list(metrics_utils.AUCCurve)))
        if isinstance(summation_method, metrics_utils.AUCSummationMethod) and summation_method not in list(metrics_utils.AUCSummationMethod):
            raise ValueError('Invalid summation method: "{}". Valid options are: "{}"'.format(summation_method, list(metrics_utils.AUCSummationMethod)))
        if thresholds is not None:
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
            self._thresholds_distributed_evenly = metrics_utils.is_evenly_distributed_thresholds(np.array([0.0] + thresholds + [1.0]))
        else:
            if num_thresholds <= 1:
                raise ValueError('`num_thresholds` must be > 1.')
            self.num_thresholds = num_thresholds
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
            self._thresholds_distributed_evenly = True
        self._thresholds = np.array([0.0 - backend.epsilon()] + thresholds + [1.0 + backend.epsilon()])
        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(summation_method)
        super(AUC, self).__init__(name=name, dtype=dtype)
        self.multi_label = multi_label
        if label_weights is not None:
            label_weights = constant_op.constant(label_weights, dtype=self.dtype)
            checks = [check_ops.assert_non_negative(label_weights, message='All values of `label_weights` must be non-negative.')]
            with ops.control_dependencies(checks):
                self.label_weights = label_weights
        else:
            self.label_weights = None
        self._from_logits = from_logits
        self._built = False
        if self.multi_label:
            if num_labels:
                shape = tensor_shape.TensorShape([None, num_labels])
                self._build(shape)
        else:
            if num_labels:
                raise ValueError('`num_labels` is needed only when `multi_label` is True.')
            self._build(None)

    @property
    def thresholds(self):
        """The thresholds used for evaluating AUC."""
        return list(self._thresholds)

    def _build(self, shape):
        """Initialize TP, FP, TN, and FN tensors, given the shape of the data."""
        if self.multi_label:
            if shape.ndims != 2:
                raise ValueError('`y_true` must have rank=2 when `multi_label` is True. Found rank %s.' % shape.ndims)
            self._num_labels = shape[1]
            variable_shape = tensor_shape.TensorShape([tensor_shape.Dimension(self.num_thresholds), self._num_labels])
        else:
            variable_shape = tensor_shape.TensorShape([tensor_shape.Dimension(self.num_thresholds)])
        self._build_input_shape = shape
        self.true_positives = self.add_weight('true_positives', shape=variable_shape, initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight('true_negatives', shape=variable_shape, initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight('false_positives', shape=variable_shape, initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', shape=variable_shape, initializer=init_ops.zeros_initializer)
        if self.multi_label:
            with ops.init_scope():
                if not context.executing_eagerly():
                    backend._initialize_variables(backend._get_session())
        self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
        deps = []
        if not self._built:
            self._build(tensor_shape.TensorShape(y_pred.shape))
        if self.multi_label or self.label_weights is not None:
            shapes = [(y_true, ('N', 'L'))]
            if self.multi_label:
                shapes.extend([(self.true_positives, ('T', 'L')), (self.true_negatives, ('T', 'L')), (self.false_positives, ('T', 'L')), (self.false_negatives, ('T', 'L'))])
            if self.label_weights is not None:
                shapes.append((self.label_weights, ('L',)))
            deps = [check_ops.assert_shapes(shapes, message='Number of labels is not consistent.')]
        label_weights = None if self.multi_label else self.label_weights
        if self._from_logits:
            y_pred = activations.sigmoid(y_pred)
        with ops.control_dependencies(deps):
            return metrics_utils.update_confusion_matrix_variables({metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives, metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives, metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives, metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives}, y_true, y_pred, self._thresholds, thresholds_distributed_evenly=self._thresholds_distributed_evenly, sample_weight=sample_weight, multi_label=self.multi_label, label_weights=label_weights)

    def interpolate_pr_auc(self):
        """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

    https://www.biostat.wisc.edu/~page/rocpr.pdf

    Note here we derive & use a closed formula not present in the paper
    as follows:

      Precision = TP / (TP + FP) = TP / P

    Modeling all of TP (true positive), FP (false positive) and their sum
    P = TP + FP (predicted positive) as varying linearly within each interval
    [A, B] between successive thresholds, we get

      Precision slope = dTP / dP
                      = (TP_B - TP_A) / (P_B - P_A)
                      = (TP - TP_A) / (P - P_A)
      Precision = (TP_A + slope * (P - P_A)) / P

    The area within the interval is (slope / total_pos_weight) times

      int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
      int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

    where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

      int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

    Bringing back the factor (slope / total_pos_weight) we'd put aside, we get

      slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

    where dTP == TP_B - TP_A.

    Note that when P_A == 0 the above calculation simplifies into

      int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

    which is really equivalent to imputing constant precision throughout the
    first bucket having >0 true positives.

    Returns:
      pr_auc: an approximation of the area under the P-R curve.
    """
        dtp = self.true_positives[:self.num_thresholds - 1] - self.true_positives[1:]
        p = self.true_positives + self.false_positives
        dp = p[:self.num_thresholds - 1] - p[1:]
        prec_slope = math_ops.div_no_nan(dtp, math_ops.maximum(dp, 0), name='prec_slope')
        intercept = self.true_positives[1:] - math_ops.multiply(prec_slope, p[1:])
        safe_p_ratio = array_ops.where(math_ops.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0), math_ops.div_no_nan(p[:self.num_thresholds - 1], math_ops.maximum(p[1:], 0), name='recall_relative_ratio'), array_ops.ones_like(p[1:]))
        pr_auc_increment = math_ops.div_no_nan(prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)), math_ops.maximum(self.true_positives[1:] + self.false_negatives[1:], 0), name='pr_auc_increment')
        if self.multi_label:
            by_label_auc = math_ops.reduce_sum(pr_auc_increment, name=self.name + '_by_label', axis=0)
            if self.label_weights is None:
                return math_ops.reduce_mean(by_label_auc, name=self.name)
            else:
                return math_ops.div_no_nan(math_ops.reduce_sum(math_ops.multiply(by_label_auc, self.label_weights)), math_ops.reduce_sum(self.label_weights), name=self.name)
        else:
            return math_ops.reduce_sum(pr_auc_increment, name='interpolate_pr_auc')

    def result(self):
        if self.curve == metrics_utils.AUCCurve.PR and self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            return self.interpolate_pr_auc()
        recall = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = math_ops.div_no_nan(self.false_positives, self.false_positives + self.true_negatives)
            x = fp_rate
            y = recall
        else:
            precision = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
            x = recall
            y = precision
        if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.0
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = math_ops.minimum(y[:self.num_thresholds - 1], y[1:])
        else:
            heights = math_ops.maximum(y[:self.num_thresholds - 1], y[1:])
        if self.multi_label:
            riemann_terms = math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights)
            by_label_auc = math_ops.reduce_sum(riemann_terms, name=self.name + '_by_label', axis=0)
            if self.label_weights is None:
                return math_ops.reduce_mean(by_label_auc, name=self.name)
            else:
                return math_ops.div_no_nan(math_ops.reduce_sum(math_ops.multiply(by_label_auc, self.label_weights)), math_ops.reduce_sum(self.label_weights), name=self.name)
        else:
            return math_ops.reduce_sum(math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights), name=self.name)

    def reset_state(self):
        if self._built:
            confusion_matrix_variables = (self.true_positives, self.true_negatives, self.false_positives, self.false_negatives)
            if self.multi_label:
                backend.batch_set_value([(v, np.zeros((self.num_thresholds, self._num_labels))) for v in confusion_matrix_variables])
            else:
                backend.batch_set_value([(v, np.zeros((self.num_thresholds,))) for v in confusion_matrix_variables])

    def get_config(self):
        if is_tensor_or_variable(self.label_weights):
            label_weights = backend.eval(self.label_weights)
        else:
            label_weights = self.label_weights
        config = {'num_thresholds': self.num_thresholds, 'curve': self.curve.value, 'summation_method': self.summation_method.value, 'thresholds': self.thresholds[1:-1], 'multi_label': self.multi_label, 'label_weights': label_weights}
        base_config = super(AUC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
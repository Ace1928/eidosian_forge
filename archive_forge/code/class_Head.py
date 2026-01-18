from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
@estimator_export('estimator.Head')
@six.add_metaclass(abc.ABCMeta)
class Head(object):
    """Interface for the head/top of a model.

  Head sits on top of the model network and handles computing the outputs of
  the network. Given logits (or output of a hidden layer), a Head knows how to
  compute predictions, loss, train_op, metrics and export outputs. It is meant
  to:

  1. Simplify writing model_fn and to make model_fn more configurable for
     Estimator.
  2. Simpilfy creating loss and metrics for the train and test loop in Eager
     execution.
  3. Support wide range of machine learning models. Since most heads can work
     with logits, they can support DNN, RNN, Wide, Wide&Deep,
     Global objectives, Gradient boosted trees and many other types
     of machine learning models.

  Common usage:
  Here is simplified model_fn to build a DNN regression model.
    ```python
    def _my_dnn_model_fn(features, labels, mode, params, config=None):
      # Optionally your callers can pass head to model_fn as a param.
      head = tf.estimator.RegressionHead(...)

      feature_columns = tf.feature_column.numeric_column(...)
      feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
      inputs = feature_layer(features)

      # Compute logits with tf.keras.layers API
      hidden_layer0 = tf.keras.layers.Dense(
          units=1000, activation="relu")(inputs)
      hidden_layer1 = tf.keras.layers.Dense(
          units=500, activation="relu")(hidden_layer0)
      logits = tf.keras.layers.Dense(
          units=head.logits_dimension, activation=None)(hidden_layer1)

      # Or use Keras model for logits computation
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Dense(units=1000, activation="relu"))
      model.add(tf.keras.layers.Dense(units=500, activation="relu"))
      model.add(tf.keras.layers.Dense(
         units=head.logits_dimension, activation=None))
      logits = model(inputs)

      return head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          optimizer=optimizer)
    ```
  """

    @abc.abstractproperty
    def name(self):
        """The name of this head.

    Returns:
      A string.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractproperty
    def logits_dimension(self):
        """Size of the last dimension of the logits `Tensor`.

    Often is the number of classes, labels, or real values to be predicted.
    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      The expected size of the `logits` tensor.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractproperty
    def loss_reduction(self):
        """One of `tf.losses.Reduction`.

    Describes how to reduce training loss over batch, such as mean or sum.

    Returns:
      The type of loss reduction used in the head.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def loss(self, labels, logits, features=None, mode=None, regularization_losses=None):
        """Returns a loss `Tensor` from provided arguments.

    Note that, the args of `features` and `mode` are most likely not used, but
    some Head implementations may require them.

    Args:
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      logits: Logits `Tensor` to be used for loss construction.
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`. To be used in case loss calculation is
        different in Train and Eval mode.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      A scalar `Tensor` representing regularized training loss used in train and
      eval.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def predictions(self, logits, keys=None):
        """Returns a `dict` of predictions from provided logits.

    Args:
      logits: Logits `Tensor` to be used for prediction construction.
      keys: A list of `string` for prediction keys. Defaults to `None`, meaning
        if not specified, predictions will be created for all the pre-defined
        valid keys in the head.

    Returns:
      A `dict` of predicted `Tensor` keyed by prediction name.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def metrics(self, regularization_losses=None):
        """Returns a `dict` of metric objects.

    Args:
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
       A `dict` of metrics keyed by string name. The value is an instance of
       `Metric` class.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def update_metrics(self, eval_metrics, features, logits, labels, mode=None, regularization_losses=None):
        """Updates metric objects and returns a `dict` of the updated metrics.

    Args:
      eval_metrics: A `dict` of metrics to be updated.
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      logits: logits `Tensor` to be used for metrics update.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      mode: Estimator's `ModeKeys`. In most cases, this arg is not used and can
        be removed in the method implementation.
      regularization_losses: A list of additional scalar losses to be added to
        the training and evaluation loss, such as regularization losses.  Note
        that, the `mode` arg is not used in the `tf.estimator.*Head`. If the
        update of the metrics doesn't rely on `mode`, it can be safely ignored
        in the method signature.

    Returns:
       A `dict` of updated metrics keyed by name. The value is an instance of
       `Metric` class.
    """
        raise NotImplementedError('Calling an abstract method.')

    def _summary_key(self, key):
        return '{}/{}'.format(key, self.name) if self.name else key

    def create_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, trainable_variables=None, train_op_fn=None, update_ops=None, regularization_losses=None):
        """Returns `EstimatorSpec` that a model_fn can return.

    It is recommended to pass all args via name.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: Logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      optimizer: An `tf.keras.optimizers.Optimizer` instance to optimize the
        loss in TRAIN mode. Namely, sets `train_op = optimizer.get_updates(loss,
        trainable_variables)`, which updates variables to minimize `loss`.
      trainable_variables: A list or tuple of `Variable` objects to update to
        minimize `loss`. In Tensorflow 1.x, by default these are the list of
        variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
        collections and GraphKeys, trainable_variables need to be passed
        explicitly here.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. By default, it is `None` in other modes. If you want to
        optimize loss yourself, you can pass `lambda _: tf.no_op()` and then use
          `EstimatorSpec.loss` to compute and apply gradients.
      update_ops: A list or tuple of update ops to be run at training time. For
        example, layers such as BatchNormalization create mean and variance
        update ops that need to be run at training time. In Tensorflow 1.x,
        these are thrown into an UPDATE_OPS collection. As Tensorflow 2.x
        doesn't have collections, update_ops need to be passed explicitly here.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      `EstimatorSpec`.
    """
        try:
            tpu_estimator_spec = self._create_tpu_estimator_spec(features=features, mode=mode, logits=logits, labels=labels, optimizer=optimizer, trainable_variables=trainable_variables, train_op_fn=train_op_fn, update_ops=update_ops, regularization_losses=regularization_losses)
            return tpu_estimator_spec.as_estimator_spec()
        except NotImplementedError:
            raise NotImplementedError('Subclasses of Head must implement `create_estimator_spec()` or _create_tpu_estimator_spec().')

    def _create_tpu_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, trainable_variables=None, train_op_fn=None, update_ops=None, regularization_losses=None):
        """Returns `model_fn._TPUEstimatorSpec` that a model_fn can return.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: Logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      optimizer: An `tf.keras.optimizers.Optimizer` instance to optimize the
        loss in TRAIN mode. Namely, sets `train_op = optimizer.get_updates(loss,
        trainable_variables)`, which updates variables to minimize `loss`.
      trainable_variables: A list or tuple of `Variable` objects to update to
        minimize `loss`. In Tensorflow 1.x, by default these are the list of
        variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
        collections and GraphKeys, trainable_variables need to be passed
        explicitly here.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. By default, it is `None` in other modes. If you want to
        optimize loss yourself, you can pass `lambda _: tf.no_op()` and then use
          `EstimatorSpec.loss` to compute and apply gradients.
      update_ops: A list or tuple of update ops to be run at training time. For
        example, layers such as BatchNormalization create mean and variance
        update ops that need to be run at training time. In Tensorflow 1.x,
        these are thrown into an UPDATE_OPS collection. As Tensorflow 2.x
        doesn't have collections, update_ops need to be passed explicitly here.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      A `model_fn._TPUEstimatorSpec' instance.
    """
        raise NotImplementedError('TPUEstimatorSpec not available for this model head.')
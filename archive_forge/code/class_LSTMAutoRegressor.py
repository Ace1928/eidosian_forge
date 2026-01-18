from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.timeseries import ar_model
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as ts_head_lib
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries import state_management
from tensorflow_estimator.python.estimator.export import export_lib
class LSTMAutoRegressor(TimeSeriesRegressor):
    """An Estimator for an LSTM autoregressive model.

  LSTMAutoRegressor is a window-based model, inputting fixed windows of length
  `input_window_size` and outputting fixed windows of length
  `output_window_size`. These two parameters must add up to the window_size
  of data returned by the `input_fn`.

  Each periodicity in the `periodicities` arg is divided by the `num_timesteps`
  into timesteps that are represented as time features added to the model.

  A good heuristic for picking an appropriate periodicity for a given data set
  would be the length of cycles in the data. For example, energy usage in a
  home is typically cyclic each day. If the time feature in a home energy
  usage dataset is in the unit of hours, then 24 would be an appropriate
  periodicity. Similarly, a good heuristic for `num_timesteps` is how often the
  data is expected to change within the cycle. For the aforementioned home
  energy usage dataset and periodicity of 24, then 48 would be a reasonable
  value if usage is expected to change every half hour.

  Each feature's value for a given example with time t is the difference
  between t and the start of the timestep it falls under. If it doesn't fall
  under a feature's associated timestep, then that feature's value is zero.

  For example: if `periodicities` = (9, 12) and `num_timesteps` = 3, then 6
  features would be added to the model, 3 for periodicity 9 and 3 for
  periodicity 12.

  For an example data point where t = 17:
  - It's in the 3rd timestep for periodicity 9 (2nd period is 9-18 and 3rd
    timestep is 15-18)
  - It's in the 2nd timestep for periodicity 12 (2nd period is 12-24 and
    2nd timestep is between 16-20).

  Therefore the 6 added features for this row with t = 17 would be:

  # Feature name (periodicity#_timestep#), feature value
  P9_T1, 0 # not in first timestep
  P9_T2, 0 # not in second timestep
  P9_T3, 2 # 17 - 15 since 15 is the start of the 3rd timestep
  P12_T1, 0 # not in first timestep
  P12_T2, 1 # 17 - 16 since 16 is the start of the 2nd timestep
  P12_T3, 0 # not in third timestep

  Example Code:

  ```python
  extra_feature_columns = (
      feature_column.numeric_column("exogenous_variable"),
  )

  estimator = LSTMAutoRegressor(
      periodicities=10,
      input_window_size=10,
      output_window_size=5,
      model_dir="/path/to/model/dir",
      num_features=1,
      extra_feature_columns=extra_feature_columns,
      num_timesteps=50,
      num_units=10,
      optimizer=tf.train.ProximalAdagradOptimizer(...))

  # Input builders
  def input_fn_train():
    return {
      "times": tf.range(15)[None, :],
      "values": tf.random_normal(shape=[1, 15, 1])
    }
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval():
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)

  def input_fn_predict():
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```
  """

    def __init__(self, periodicities, input_window_size, output_window_size, model_dir=None, num_features=1, extra_feature_columns=None, num_timesteps=10, loss=ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS, num_units=128, optimizer='Adam', config=None):
        """Initialize the Estimator.

    Args:
      periodicities: periodicities of the input data, in the same units as the
        time feature (for example 24 if feeding hourly data with a daily
        periodicity, or 60 * 24 if feeding minute-level data with daily
        periodicity). Note this can be a single value or a list of values for
        multiple periodicities.
      input_window_size: Number of past time steps of data to look at when doing
        the regression.
      output_window_size: Number of future time steps to predict. Note that
        setting this value to > 1 empirically seems to give a better fit.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      num_features: The dimensionality of the time series (default value is one
        for univariate, more than one for multivariate).
      extra_feature_columns: A list of `tf.feature_column`s (for example
        `tf.feature_column.embedding_column`) corresponding to features which
        provide extra information to the model but are not part of the series to
        be predicted.
      num_timesteps: Number of buckets into which to divide (time %
        periodicity). This value multiplied by the number of periodicities is
        the number of time features added to the model.
      loss: Loss function to use for training. Currently supported values are
        SQUARED_LOSS and NORMAL_LIKELIHOOD_LOSS. Note that for
        NORMAL_LIKELIHOOD_LOSS, we train the covariance term as well. For
        SQUARED_LOSS, the evaluation loss is reported based on un-scaled
        observations and predictions, while the training loss is computed on
        normalized data.
      num_units: The size of the hidden state in the encoder and decoder LSTM
        cells.
      optimizer: string, `tf.train.Optimizer` object, or callable that defines
        the optimizer algorithm to use for training. Defaults to the Adam
        optimizer with a learning rate of 0.01.
      config: Optional `estimator.RunConfig` object to configure the runtime
        settings.
    """
        optimizer = optimizers.get_optimizer_instance(optimizer, learning_rate=0.01)
        model = ar_model.ARModel(periodicities=periodicities, input_window_size=input_window_size, output_window_size=output_window_size, num_features=num_features, exogenous_feature_columns=extra_feature_columns, num_time_buckets=num_timesteps, loss=loss, prediction_model_factory=functools.partial(ar_model.LSTMPredictionModel, num_units=num_units))
        state_manager = state_management.FilteringOnlyStateManager()
        super(LSTMAutoRegressor, self).__init__(model=model, state_manager=state_manager, optimizer=optimizer, model_dir=model_dir, config=config, head_type=ts_head_lib.OneShotPredictionHead)
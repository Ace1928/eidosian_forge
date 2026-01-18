from tensorflow.python.eager import backprop
from tensorflow.python.training import optimizer as optimizer_lib
class StandardInputStep(Step):
    """Step with a standard implementation of input handling.

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
  """

    def __init__(self, dataset_fn, distribution):
        super(StandardInputStep, self).__init__(distribution)
        self._iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())

    def initialize(self):
        return self._iterator.initializer
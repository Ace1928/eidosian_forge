from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export
Returns the Tensorflow eager tensor.

  The returned tensor uses the memory shared by dlpack capsules from other
  framework.

    ```python
    a = tf.experimental.dlpack.from_dlpack(dlcapsule)
    # `a` uses the memory shared by dlpack
    ```

  Args:
    dlcapsule: A PyCapsule named as dltensor

  Returns:
    A Tensorflow eager tensor
  
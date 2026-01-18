from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (

    Generates a sample ndarray or dict (str -> ndarray)
    from the input type 'x' for keras ``fit`` or ``fit_generator``

    Args:
        input_training_data: Keras input function used for ``fit`` or ``fit_generator`` methods.

    Returns:
        a slice of type ndarray or
        dict (str -> ndarray) limited to
        ``mlflow.utils.autologging_utils.INPUT_EXAMPLE_SAMPLE_ROWS``.
        Throws ``MlflowException`` exception, if input_training_data is unsupported.
        Returns `None` if the type of input_training_data is unsupported.
    
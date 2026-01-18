from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (
def _extract_input_example_from_tensor_or_ndarray(input_features: Union[tensorflow.Tensor, np.ndarray]) -> np.ndarray:
    """
    Extracts first `INPUT_EXAMPLE_SAMPLE_ROWS` from the next_input, which can either be of
    numpy array or tensor type.

    Args:
        input_features: an input of type `np.ndarray` or `tensorflow.Tensor`

    Returns:
        A slice (of limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of type `np.ndarray`.
        Returns `None` if the type of `input_features` is unsupported.

    Examples
    --------
    when next_input is nd.array:
    >>> input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> _extract_input_example_from_tensor_or_ndarray(input_data)
    array([1, 2, 3, 4, 5])


    when next_input is tensorflow.Tensor:
    >>> input_data = tensorflow.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> _extract_input_example_from_tensor_or_ndarray(input_data)
    array([1, 2, 3, 4, 5])
    """
    input_feature_slice = None
    if isinstance(input_features, tensorflow.Tensor):
        input_feature_slice = input_features.numpy()[0:INPUT_EXAMPLE_SAMPLE_ROWS]
    elif isinstance(input_features, np.ndarray):
        input_feature_slice = input_features[0:INPUT_EXAMPLE_SAMPLE_ROWS]
    return input_feature_slice
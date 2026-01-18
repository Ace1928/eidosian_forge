from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (
def _extract_input_example_from_batched_tf_dataset(dataset: tensorflow.data.Dataset) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extracts sample feature tensors from the input dataset as numpy array.
    Input Dataset's tensors must contain tuple of (features, labels) that are
    used for tensorflow/keras train or fit methods


    Args:
        dataset: a tensorflow batched/unbatched dataset representing tuple of (features, labels)

    Returns:
        a numpy array of length `INPUT_EXAMPLE_SAMPLE_ROWS`
        Returns `None` if the type of `dataset` slices are unsupported.

    Examples
    --------
    >>> input_dataset = tensorflow.data.Dataset.from_tensor_slices(
    ...     (
    ...         {
    ...             "SepalLength": np.array(list(range(0, 20))),
    ...             "SepalWidth": np.array(list(range(0, 20))),
    ...             "PetalLength": np.array(list(range(0, 20))),
    ...             "PetalWidth": np.array(list(range(0, 20))),
    ...         },
    ...         np.array(list(range(0, 20))),
    ...     )
    ... ).batch(10)
    >>> _extract_input_example_from_batched_tf_dataset(input_dataset)
    {'SepalLength': array([0, 1, 2, 3, 4]),
    'SepalWidth': array([0, 1, 2, 3, 4]),
    'PetalLength': array([0, 1, 2, 3, 4]),
    'PetalWidth': array([0, 1, 2, 3, 4])}

    """
    limited_df_iter = list(dataset.take(INPUT_EXAMPLE_SAMPLE_ROWS))
    first_batch = limited_df_iter[0]
    input_example_slice = None
    if isinstance(first_batch, tuple):
        features = first_batch[0]
        if isinstance(features, dict):
            input_example_slice = _extract_sample_numpy_dict(features)
        elif isinstance(features, (np.ndarray, tensorflow.Tensor)):
            input_example_slice = _extract_input_example_from_tensor_or_ndarray(features)
    return input_example_slice
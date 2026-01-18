from keras.src import backend
from keras.src import ops
Cast a list of tensors to a common dtype.

    If any tensor is floating-point, they will all be casted to the most-precise
    floating-point dtype. Otherwise the tensors are not casted.

    Args:
        tensors: A list of tensors.

    Returns:
        Same list, casted to a common dtype.
    
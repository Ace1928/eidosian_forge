from copy import deepcopy
from typing import Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available
def _check_and_preprocess_points(self, input_points=None, input_labels=None, input_boxes=None):
    """
        Check and preprocesses the 2D points, labels and bounding boxes. It checks if the input is valid and if they
        are, it converts the coordinates of the points and bounding boxes. If a user passes directly a `torch.Tensor`,
        it is converted to a `numpy.ndarray` and then to a `list`.
        """
    if input_points is not None:
        if hasattr(input_points, 'numpy'):
            input_points = input_points.numpy().tolist()
        if not isinstance(input_points, list) or not isinstance(input_points[0], list):
            raise ValueError('Input points must be a list of list of floating points.')
        input_points = [np.array(input_point) for input_point in input_points]
    else:
        input_points = None
    if input_labels is not None:
        if hasattr(input_labels, 'numpy'):
            input_labels = input_labels.numpy().tolist()
        if not isinstance(input_labels, list) or not isinstance(input_labels[0], list):
            raise ValueError('Input labels must be a list of list integers.')
        input_labels = [np.array(label) for label in input_labels]
    else:
        input_labels = None
    if input_boxes is not None:
        if hasattr(input_boxes, 'numpy'):
            input_boxes = input_boxes.numpy().tolist()
        if not isinstance(input_boxes, list) or not isinstance(input_boxes[0], list) or (not isinstance(input_boxes[0][0], list)):
            raise ValueError('Input boxes must be a list of list of list of floating points.')
        input_boxes = [np.array(box).astype(np.float32) for box in input_boxes]
    else:
        input_boxes = None
    return (input_points, input_labels, input_boxes)
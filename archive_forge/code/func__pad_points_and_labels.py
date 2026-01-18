from copy import deepcopy
from typing import Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available
def _pad_points_and_labels(self, input_points, input_labels):
    """
        The method pads the 2D points and labels to the maximum number of points in the batch.
        """
    expected_nb_points = max([point.shape[0] for point in input_points])
    processed_input_points = []
    for i, point in enumerate(input_points):
        if point.shape[0] != expected_nb_points:
            point = np.concatenate([point, np.zeros((expected_nb_points - point.shape[0], 2)) + self.point_pad_value], axis=0)
            input_labels[i] = np.append(input_labels[i], [self.point_pad_value])
        processed_input_points.append(point)
    input_points = processed_input_points
    return (input_points, input_labels)
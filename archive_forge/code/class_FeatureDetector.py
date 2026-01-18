import numpy as np
from ..util import img_as_float
from .._shared.utils import (
class FeatureDetector:

    def __init__(self):
        self.keypoints_ = np.array([])

    def detect(self, image):
        """Detect keypoints in image.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        raise NotImplementedError()
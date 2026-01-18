import numpy as np
from . import _hoghistogram
from .._shared import utils

    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    
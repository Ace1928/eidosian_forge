import torch
from torch import Tensor

    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) format.
    
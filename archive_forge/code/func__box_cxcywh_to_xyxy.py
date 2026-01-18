import torch
from torch import Tensor
def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes
import numbers
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
@torch.jit.unused
def _parse_fill(fill: Optional[Union[float, List[float], Tuple[float, ...]]], img: Image.Image, name: str='fillcolor') -> Dict[str, Optional[Union[float, List[float], Tuple[float, ...]]]]:
    num_channels = get_image_num_channels(img)
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_channels > 1:
        fill = tuple([fill] * num_channels)
    if isinstance(fill, (list, tuple)):
        if len(fill) == 1:
            fill = fill * num_channels
        elif len(fill) != num_channels:
            msg = "The number of elements in 'fill' does not match the number of channels of the image ({} != {})"
            raise ValueError(msg.format(len(fill), num_channels))
        fill = tuple(fill)
    if img.mode != 'F':
        if isinstance(fill, (list, tuple)):
            fill = tuple((int(x) for x in fill))
        else:
            fill = int(fill)
    return {name: fill}
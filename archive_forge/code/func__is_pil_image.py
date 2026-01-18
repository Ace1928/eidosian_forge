import numbers
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
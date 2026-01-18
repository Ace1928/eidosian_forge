import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms import _functional_tensor as _FT
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
from ._utils import _get_fill, _setup_fill_arg, check_type, is_pure_tensor
def _apply_image_or_video_transform(self, image: ImageOrVideo, transform_id: str, magnitude: float, interpolation: Union[InterpolationMode, int], fill: Dict[Union[Type, str], _FillTypeJIT]) -> ImageOrVideo:
    fill_ = _get_fill(fill, type(image))
    if transform_id == 'Identity':
        return image
    elif transform_id == 'ShearX':
        return F.affine(image, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(math.atan(magnitude)), 0.0], interpolation=interpolation, fill=fill_, center=[0, 0])
    elif transform_id == 'ShearY':
        return F.affine(image, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(math.atan(magnitude))], interpolation=interpolation, fill=fill_, center=[0, 0])
    elif transform_id == 'TranslateX':
        return F.affine(image, angle=0.0, translate=[int(magnitude), 0], scale=1.0, interpolation=interpolation, shear=[0.0, 0.0], fill=fill_)
    elif transform_id == 'TranslateY':
        return F.affine(image, angle=0.0, translate=[0, int(magnitude)], scale=1.0, interpolation=interpolation, shear=[0.0, 0.0], fill=fill_)
    elif transform_id == 'Rotate':
        return F.rotate(image, angle=magnitude, interpolation=interpolation, fill=fill_)
    elif transform_id == 'Brightness':
        return F.adjust_brightness(image, brightness_factor=1.0 + magnitude)
    elif transform_id == 'Color':
        return F.adjust_saturation(image, saturation_factor=1.0 + magnitude)
    elif transform_id == 'Contrast':
        return F.adjust_contrast(image, contrast_factor=1.0 + magnitude)
    elif transform_id == 'Sharpness':
        return F.adjust_sharpness(image, sharpness_factor=1.0 + magnitude)
    elif transform_id == 'Posterize':
        return F.posterize(image, bits=int(magnitude))
    elif transform_id == 'Solarize':
        bound = _FT._max_value(image.dtype) if isinstance(image, torch.Tensor) else 255.0
        return F.solarize(image, threshold=bound * magnitude)
    elif transform_id == 'AutoContrast':
        return F.autocontrast(image)
    elif transform_id == 'Equalize':
        return F.equalize(image)
    elif transform_id == 'Invert':
        return F.invert(image)
    else:
        raise ValueError(f'No transform available for {transform_id}')
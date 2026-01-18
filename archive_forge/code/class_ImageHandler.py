import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
class ImageHandler:
    """
    Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """

    def __init__(self, imagespec):
        assert imagespec in list(imagespecs.keys()), f'unknown image specification: {imagespec}'
        self.imagespec = imagespec.lower()

    def __call__(self, extension, data):
        if extension.lower() not in 'jpg jpeg png ppm pgm pbm pnm'.split():
            return None
        try:
            import numpy as np
        except ImportError as e:
            raise ModuleNotFoundError('Package `numpy` is required to be installed for default image decoder.Please use `pip install numpy` to install the package') from e
        try:
            import PIL.Image
        except ImportError as e:
            raise ModuleNotFoundError('Package `PIL` is required to be installed for default image decoder.Please use `pip install Pillow` to install the package') from e
        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
            if atype == 'pil':
                return img
            elif atype == 'numpy':
                result = np.asarray(img)
                assert result.dtype == np.uint8, f'numpy image array should be type uint8, but got {result.dtype}'
                if etype == 'uint8':
                    return result
                else:
                    return result.astype('f') / 255.0
            elif atype == 'torch':
                result = np.asarray(img)
                assert result.dtype == np.uint8, f'numpy image array should be type uint8, but got {result.dtype}'
                if etype == 'uint8':
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result)
                else:
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result) / 255.0
            return None
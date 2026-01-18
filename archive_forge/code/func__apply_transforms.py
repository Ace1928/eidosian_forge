import sys
import warnings
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast
import numpy as np
from PIL import ExifTags, GifImagePlugin, Image, ImageSequence, UnidentifiedImageError
from PIL import __version__ as pil_version  # type: ignore
from ..core.request import URI_BYTES, InitializationError, IOMode, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike
def _apply_transforms(self, image, mode, rotate, apply_gamma, writeable_output) -> np.ndarray:
    if mode is not None:
        image = image.convert(mode)
    elif image.mode == 'P':
        image = image.convert(image.palette.mode)
    elif image.format == 'PNG' and image.mode == 'I':
        major, minor, patch = pillow_version()
        if sys.byteorder == 'little':
            desired_mode = 'I;16'
        else:
            desired_mode = 'I;16B'
        if major < 10:
            warnings.warn("Loading 16-bit (uint16) PNG as int32 due to limitations in pillow's PNG decoder. This will be fixed in a future version of pillow which will make this warning dissapear.", UserWarning)
        elif minor < 1:
            image.mode = desired_mode
        else:
            image = image.convert(desired_mode)
    image = np.asarray(image)
    meta = self.metadata(index=self._image.tell(), exclude_applied=False)
    if rotate and 'Orientation' in meta:
        transformation = _exif_orientation_transform(meta['Orientation'], self._image.mode)
        image = transformation(image)
    if apply_gamma and 'gamma' in meta:
        gamma = float(meta['gamma'])
        scale = float(65536 if image.dtype == np.uint16 else 255)
        gain = 1.0
        image = (image / scale) ** gamma * scale * gain + 0.4999
        image = np.round(image).astype(np.uint8)
    if writeable_output and (not image.flags['WRITEABLE']):
        image = np.array(image)
    return image
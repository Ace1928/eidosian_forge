import math
import struct
import zlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ._utils import (
from .constants import CcittFaxDecodeParameters as CCITT
from .constants import ColorSpaces
from .constants import FilterTypeAbbreviations as FTA
from .constants import FilterTypes as FT
from .constants import ImageAttributes as IA
from .constants import LzwFilterParameters as LZW
from .constants import StreamAttributes as SA
from .errors import DeprecationError, PdfReadError, PdfStreamError
from .generic import (
def _xobj_to_image(x_object_obj: Dict[str, Any]) -> Tuple[Optional[str], bytes, Any]:
    """
    Users need to have the pillow package installed.

    It's unclear if pypdf will keep this function here, hence it's private.
    It might get removed at any point.

    Args:
      x_object_obj:

    Returns:
        Tuple[file extension, bytes, PIL.Image.Image]
    """
    from ._xobj_image_helpers import Image, _get_imagemode, _handle_flate, _handle_jpx, mode_str_type
    if hasattr(x_object_obj, 'indirect_reference') and x_object_obj is None:
        obj_as_text = x_object_obj.indirect_reference.__repr__()
    else:
        obj_as_text = x_object_obj.__repr__()
    size = (x_object_obj[IA.WIDTH], x_object_obj[IA.HEIGHT])
    data = x_object_obj.get_data()
    if isinstance(data, str):
        data = data.encode()
    colors = x_object_obj.get('/Colors', 1)
    color_space: Any = x_object_obj.get('/ColorSpace', NullObject()).get_object()
    if isinstance(color_space, list) and len(color_space) == 1:
        color_space = color_space[0].get_object()
    if IA.COLOR_SPACE in x_object_obj and x_object_obj[IA.COLOR_SPACE] == ColorSpaces.DEVICE_RGB:
        mode: mode_str_type = 'RGB'
    if x_object_obj.get('/BitsPerComponent', 8) < 8:
        mode, invert_color = _get_imagemode(f'{x_object_obj.get('/BitsPerComponent', 8)}bit', 0, '')
    else:
        mode, invert_color = _get_imagemode(color_space, 2 if colors == 1 and (not isinstance(color_space, NullObject) and 'Gray' not in color_space) else colors, '')
    extension = None
    alpha = None
    filters = x_object_obj.get(SA.FILTER, NullObject()).get_object()
    lfilters = filters[-1] if isinstance(filters, list) else filters
    if lfilters in (FT.FLATE_DECODE, FT.RUN_LENGTH_DECODE):
        img, image_format, extension, _ = _handle_flate(size, data, mode, color_space, colors, obj_as_text)
    elif lfilters in (FT.LZW_DECODE, FT.ASCII_85_DECODE, FT.CCITT_FAX_DECODE):
        if x_object_obj[SA.FILTER] in [[FT.LZW_DECODE], [FT.CCITT_FAX_DECODE]]:
            extension = '.tiff'
            image_format = 'TIFF'
        else:
            extension = '.png'
            image_format = 'PNG'
        img = Image.open(BytesIO(data), formats=('TIFF', 'PNG'))
    elif lfilters == FT.DCT_DECODE:
        img, image_format, extension = (Image.open(BytesIO(data)), 'JPEG', '.jpg')
    elif lfilters == FT.JPX_DECODE:
        img, image_format, extension, invert_color = _handle_jpx(size, data, mode, color_space, colors)
    elif lfilters == FT.CCITT_FAX_DECODE:
        img, image_format, extension, invert_color = (Image.open(BytesIO(data), formats=('TIFF',)), 'TIFF', '.tiff', False)
    elif mode == 'CMYK':
        img, image_format, extension, invert_color = (Image.frombytes(mode, size, data), 'TIFF', '.tif', False)
    elif mode == '':
        raise PdfReadError(f'ColorSpace field not found in {x_object_obj}')
    else:
        img, image_format, extension, invert_color = (Image.frombytes(mode, size, data), 'PNG', '.png', False)
    decode = x_object_obj.get(IA.DECODE, [1.0, 0.0] * len(img.getbands()) if img.mode == 'CMYK' and lfilters in (FT.DCT_DECODE, FT.JPX_DECODE) or (invert_color and img.mode == 'L') else None)
    if isinstance(color_space, ArrayObject) and color_space[0].get_object() == '/Indexed':
        decode = None
    if decode is not None and (not all((decode[i] == i % 2 for i in range(len(decode))))):
        lut: List[int] = []
        for i in range(0, len(decode), 2):
            dmin = decode[i]
            dmax = decode[i + 1]
            lut.extend((round(255.0 * (j / 255.0 * (dmax - dmin) + dmin)) for j in range(256)))
        img = img.point(lut)
    if IA.S_MASK in x_object_obj:
        alpha = _xobj_to_image(x_object_obj[IA.S_MASK])[2]
        if img.size != alpha.size:
            logger_warning(f'image and mask size not matching: {obj_as_text}', __name__)
        else:
            if alpha.mode != 'L':
                alpha = alpha.convert('L')
            if img.mode == 'P':
                img = img.convert('RGB')
            elif img.mode == '1':
                img = img.convert('L')
            img.putalpha(alpha)
        if 'JPEG' in image_format:
            extension = '.jp2'
            image_format = 'JPEG2000'
        else:
            extension = '.png'
            image_format = 'PNG'
    img_byte_arr = BytesIO()
    try:
        img.save(img_byte_arr, format=image_format)
    except OSError:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format=image_format)
    data = img_byte_arr.getvalue()
    try:
        img = Image.open(BytesIO(data))
    except Exception:
        img = None
    return (extension, data, img)
import sys
from io import BytesIO
from typing import Any, List, Tuple, Union, cast
from ._utils import check_if_whitespace_only, logger_warning
from .constants import ColorSpaces
from .errors import PdfReadError
from .generic import (
def _get_imagemode(color_space: Union[str, List[Any], Any], color_components: int, prev_mode: mode_str_type, depth: int=0) -> Tuple[mode_str_type, bool]:
    """
    Returns
        Image mode not taking into account mask(transparency)
        ColorInversion is required (like for some DeviceCMYK)
    """
    if depth > MAX_IMAGE_MODE_NESTING_DEPTH:
        raise PdfReadError('Color spaces nested too deep. If required, consider increasing MAX_IMAGE_MODE_NESTING_DEPTH.')
    if isinstance(color_space, NullObject):
        return ('', False)
    if isinstance(color_space, str):
        pass
    elif not isinstance(color_space, list):
        raise PdfReadError('Cannot interpret colorspace', color_space)
    elif color_space[0].startswith('/Cal'):
        color_space = '/Device' + color_space[0][4:]
    elif color_space[0] == '/ICCBased':
        icc_profile = color_space[1].get_object()
        color_components = cast(int, icc_profile['/N'])
        color_space = icc_profile.get('/Alternate', '')
    elif color_space[0] == '/Indexed':
        color_space = color_space[1]
        if isinstance(color_space, IndirectObject):
            color_space = color_space.get_object()
        mode2, invert_color = _get_imagemode(color_space, color_components, prev_mode, depth + 1)
        if mode2 in ('RGB', 'CMYK'):
            mode2 = 'P'
        return (mode2, invert_color)
    elif color_space[0] == '/Separation':
        color_space = color_space[2]
        if isinstance(color_space, IndirectObject):
            color_space = color_space.get_object()
        mode2, invert_color = _get_imagemode(color_space, color_components, prev_mode, depth + 1)
        return (mode2, True)
    elif color_space[0] == '/DeviceN':
        original_color_space = color_space
        color_components = len(color_space[1])
        color_space = color_space[2]
        if isinstance(color_space, IndirectObject):
            color_space = color_space.get_object()
        if color_space == '/DeviceCMYK' and color_components == 1:
            if original_color_space[1][0] != '/Black':
                logger_warning(f'Color {original_color_space[1][0]} converted to Gray. Please share PDF with pypdf dev team', __name__)
            return ('L', True)
        mode2, invert_color = _get_imagemode(color_space, color_components, prev_mode, depth + 1)
        return (mode2, invert_color)
    mode_map = {'1bit': '1', '/DeviceGray': 'L', 'palette': 'P', '/DeviceRGB': 'RGB', '/DeviceCMYK': 'CMYK', '2bit': '2bits', '4bit': '4bits'}
    mode: mode_str_type = mode_map.get(color_space) or list(mode_map.values())[color_components] or prev_mode
    return (mode, mode == 'CMYK')
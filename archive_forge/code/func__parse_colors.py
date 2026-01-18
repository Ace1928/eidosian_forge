import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
def _parse_colors(colors: Union[None, str, Tuple[int, int, int], List[Union[str, Tuple[int, int, int]]]], *, num_objects: int) -> List[Tuple[int, int, int]]:
    """
    Parses a specification of colors for a set of objects.

    Args:
        colors: A specification of colors for the objects. This can be one of the following:
            - None: to generate a color palette automatically.
            - A list of colors: where each color is either a string (specifying a named color) or an RGB tuple.
            - A string or an RGB tuple: to use the same color for all objects.

            If `colors` is a tuple, it should be a 3-tuple specifying the RGB values of the color.
            If `colors` is a list, it should have at least as many elements as the number of objects to color.

        num_objects (int): The number of objects to color.

    Returns:
        A list of 3-tuples, specifying the RGB values of the colors.

    Raises:
        ValueError: If the number of colors in the list is less than the number of objects to color.
                    If `colors` is not a list, tuple, string or None.
    """
    if colors is None:
        colors = _generate_color_palette(num_objects)
    elif isinstance(colors, list):
        if len(colors) < num_objects:
            raise ValueError(f'Number of colors must be equal or larger than the number of objects, but got {len(colors)} < {num_objects}.')
    elif not isinstance(colors, (tuple, str)):
        raise ValueError('`colors` must be a tuple or a string, or a list thereof, but got {colors}.')
    elif isinstance(colors, tuple) and len(colors) != 3:
        raise ValueError('If passed as tuple, colors should be an RGB triplet, but got {colors}.')
    else:
        colors = [colors] * num_objects
    return [ImageColor.getrgb(color) if isinstance(color, str) else color for color in colors]
import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import warn
def _generate_rectangle_mask(point, image, shape, random):
    """Generate a mask for a filled rectangle shape.

    The height and width of the rectangle are generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape
        is placed.
    shape : tuple
        The minimum and maximum size of the shape to fit.
    random : `numpy.random.Generator`

        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.

    """
    available_width = min(image[1] - point[1], shape[1]) - shape[0]
    available_height = min(image[0] - point[0], shape[1]) - shape[0]
    r = shape[0] + random.integers(max(1, available_height)) - 1
    c = shape[0] + random.integers(max(1, available_width)) - 1
    rectangle = draw_polygon([point[0], point[0] + r, point[0] + r, point[0]], [point[1], point[1], point[1] + c, point[1] + c])
    label = ('rectangle', ((point[0], point[0] + r + 1), (point[1], point[1] + c + 1)))
    return (rectangle, label)
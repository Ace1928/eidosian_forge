import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import warn
def _generate_circle_mask(point, image, shape, random):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
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
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] // 2.0
    max_radius = shape[1] // 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius) - min_radius
    if available_radius < 0:
        raise ArithmeticError('cannot fit shape to image')
    radius = int(min_radius + random.integers(max(1, available_radius)))
    disk = draw_disk((point[0], point[1]), radius)
    label = ('circle', ((point[0] - radius + 1, point[0] + radius), (point[1] - radius + 1, point[1] + radius)))
    return (disk, label)
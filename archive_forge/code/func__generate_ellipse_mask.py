import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import warn
def _generate_ellipse_mask(point, image, shape, random):
    """Generate a mask for a filled ellipse shape.

    The rotation, major and minor semi-axes of the ellipse are generated
    randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is
        placed.
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
        raise ValueError('size must be > 1 for ellipses')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    r_radius = random.uniform(min_radius, available_radius + 1)
    c_radius = random.uniform(min_radius, available_radius + 1)
    rotation = random.uniform(-np.pi, np.pi)
    ellipse = draw_ellipse(point[0], point[1], r_radius, c_radius, shape=image[:2], rotation=rotation)
    max_radius = math.ceil(max(r_radius, c_radius))
    min_x = np.min(ellipse[0])
    max_x = np.max(ellipse[0]) + 1
    min_y = np.min(ellipse[1])
    max_y = np.max(ellipse[1]) + 1
    label = ('ellipse', ((min_x, max_x), (min_y, max_y)))
    return (ellipse, label)
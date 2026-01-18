import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import warn
def _generate_triangle_mask(point, image, shape, random):
    """Generate a mask for a filled equilateral triangle shape.

    The length of the sides of the triangle is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of a up-pointing triangle.
    image : tuple
        The height, width and depth of the image into which the shape
        is placed.
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
        raise ValueError('dimension must be > 1 for triangles')
    available_side = min(image[1] - point[1], point[0], shape[1]) - shape[0]
    side = shape[0] + random.integers(max(1, available_side)) - 1
    triangle_height = int(np.ceil(np.sqrt(3 / 4.0) * side))
    triangle = draw_polygon([point[0], point[0] - triangle_height, point[0]], [point[1], point[1] + side // 2, point[1] + side])
    label = ('triangle', ((point[0] - triangle_height, point[0] + 1), (point[1], point[1] + side + 1)))
    return (triangle, label)
from math import sqrt
def _count_intersection(l1, l2):
    """Count intersection between two line segments defined by coordinate pairs.

    :param l1: Tuple of two coordinate pairs defining the first line segment
    :param l2: Tuple of two coordinate pairs defining the second line segment
    :type l1: tuple(float, float)
    :type l2: tuple(float, float)
    :return: Coordinates of the intersection
    :rtype: tuple(float, float)
    """
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0.0:
        if x1 == x2 == x3 == x4 == 0.0:
            return (0.0, y4)
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (x, y)
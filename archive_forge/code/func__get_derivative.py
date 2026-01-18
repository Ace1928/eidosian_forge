from math import sqrt
def _get_derivative(coordinates):
    """Get derivative of the line from (0,0) to given coordinates.

    :param coordinates: A coordinate pair
    :type coordinates: tuple(float, float)
    :return: Derivative; inf if x is zero
    :rtype: float
    """
    try:
        return coordinates[1] / coordinates[0]
    except ZeroDivisionError:
        return float('inf')
from math import sqrt
def _indexes(gumt, gdmt, gwmt, gdnt):
    """Count Understemming Index (UI), Overstemming Index (OI) and Stemming Weight (SW).

    :param gumt, gdmt, gwmt, gdnt: Global unachieved merge total (gumt),
    global desired merge total (gdmt),
    global wrongly merged total (gwmt) and
    global desired non-merge total (gdnt).
    :type gumt, gdmt, gwmt, gdnt: float
    :return: Understemming Index (UI),
    Overstemming Index (OI) and
    Stemming Weight (SW).
    :rtype: tuple(float, float, float)
    """
    try:
        ui = gumt / gdmt
    except ZeroDivisionError:
        ui = 0.0
    try:
        oi = gwmt / gdnt
    except ZeroDivisionError:
        oi = 0.0
    try:
        sw = oi / ui
    except ZeroDivisionError:
        if oi == 0.0:
            sw = float('nan')
        else:
            sw = float('inf')
    return (ui, oi, sw)
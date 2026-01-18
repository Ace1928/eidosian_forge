import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class FormatterPrettyPrint:

    def __init__(self, useMathText=True):
        self._fmt = mticker.ScalarFormatter(useMathText=useMathText, useOffset=False)
        self._fmt.create_dummy_axis()

    def __call__(self, direction, factor, values):
        return self._fmt.format_ticks(values)
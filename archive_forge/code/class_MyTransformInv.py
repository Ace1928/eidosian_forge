import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D, Transform
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
class MyTransformInv(Transform):
    input_dims = output_dims = 2

    def __init__(self, resolution):
        Transform.__init__(self)
        self._resolution = resolution

    def transform(self, ll):
        x, y = ll.T
        return np.column_stack([x, y + x])

    def inverted(self):
        return MyTransform(self._resolution)
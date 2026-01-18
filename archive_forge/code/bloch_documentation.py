import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
Saves Bloch sphere to file of type ``format`` in directory ``dirc``.

        Args:
            name (str):
                Name of saved image. Must include path and format as well.
                i.e. '/Users/Paul/Desktop/bloch.png'
                This overrides the 'format' and 'dirc' arguments.
            output (str):
                Format of output image.
            dirc (str):
                Directory for output images. Defaults to current working directory.
        
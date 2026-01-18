import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np
import pandas as pd
def _generate_hexagon(self, center: Point3D, elevation: float) -> Hexagon3D:
    logging.debug(f'Starting hexagon generation with center={center} and elevation={elevation}')
    vertices: Hexagon3D = []
    for i in range(6):
        angle_rad = 2 * math.pi / 6 * i
        x = center[0] + self.side_length * math.cos(angle_rad)
        y = center[1] + self.side_length * math.sin(angle_rad)
        vertices.append((x, y, elevation))
    logging.debug(f'Hexagon generated with vertices={vertices}')
    return vertices
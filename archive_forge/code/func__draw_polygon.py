from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require
@require('matplotlib', '>=3.3')
def _draw_polygon(ax, vertices, alpha=0.4):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    polygon = Polygon(vertices, closed=True)
    p = PatchCollection([polygon], match_original=True, alpha=alpha)
    polygon_object = ax.add_collection(p)
    plt.draw()
    return polygon_object
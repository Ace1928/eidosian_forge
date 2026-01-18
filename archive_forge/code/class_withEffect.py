from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
class withEffect(effect_class):

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        super().draw_path(renderer, gc, tpath, affine, rgbFace)
        renderer.draw_path(gc, tpath, affine, rgbFace)
import numpy as np
from matplotlib import cbook
from .backend_agg import RendererAgg
from matplotlib._tight_bbox import process_figure_for_rasterizing

        Exit "raster" mode.  All of the drawing that was done since
        the last `start_rasterizing` call will be copied to the
        vector backend by calling draw_image.
        
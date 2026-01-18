import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
class UnsnappedMarkerStyle(markers.MarkerStyle):
    """
    A MarkerStyle where the snap threshold is force-disabled.

    This is used to compare to polygon/star/asterisk markers which do not have
    any snap threshold set.
    """

    def _recache(self):
        super()._recache()
        self._snap_threshold = None
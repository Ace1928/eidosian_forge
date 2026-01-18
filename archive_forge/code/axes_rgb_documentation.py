from types import MethodType
import numpy as np
from .axes_divider import make_axes_locatable, Size
from .mpl_axes import Axes, SimpleAxisArtist

        Create the four images {rgb, r, g, b}.

        Parameters
        ----------
        r, g, b : array-like
            The red, green, and blue arrays.
        **kwargs
            Forwarded to `~.Axes.imshow` calls for the four images.

        Returns
        -------
        rgb : `~matplotlib.image.AxesImage`
        r : `~matplotlib.image.AxesImage`
        g : `~matplotlib.image.AxesImage`
        b : `~matplotlib.image.AxesImage`
        
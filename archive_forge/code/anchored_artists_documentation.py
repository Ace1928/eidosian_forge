from matplotlib import _api, transforms
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
from matplotlib.patches import (Rectangle, Ellipse, ArrowStyle,
from matplotlib.text import TextPath

        Draw two perpendicular arrows to indicate directions.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transAxes`.
        label_x, label_y : str
            Label text for the x and y arrows
        length : float, default: 0.15
            Length of the arrow, given in coordinates of *transform*.
        fontsize : float, default: 0.08
            Size of label strings, given in coordinates of *transform*.
        loc : str, default: 'upper left'
            Location of the arrow.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        angle : float, default: 0
            The angle of the arrows in degrees.
        aspect_ratio : float, default: 1
            The ratio of the length of arrow_x and arrow_y.
            Negative numbers can be used to change the direction.
        pad : float, default: 0.4
            Padding around the labels and arrows, in fraction of the font size.
        borderpad : float, default: 0.4
            Border padding, in fraction of the font size.
        frameon : bool, default: False
            If True, draw a box around the arrows and labels.
        color : str, default: 'white'
            Color for the arrows and labels.
        alpha : float, default: 1
            Alpha values of the arrows and labels
        sep_x, sep_y : float, default: 0.01 and 0 respectively
            Separation between the arrows and labels in coordinates of
            *transform*.
        fontproperties : `~matplotlib.font_manager.FontProperties`, optional
            Font properties for the label text.
        back_length : float, default: 0.15
            Fraction of the arrow behind the arrow crossing.
        head_width : float, default: 10
            Width of arrow head, sent to `.ArrowStyle`.
        head_length : float, default: 15
            Length of arrow head, sent to `.ArrowStyle`.
        tail_width : float, default: 2
            Width of arrow tail, sent to `.ArrowStyle`.
        text_props, arrow_props : dict
            Properties of the text and arrows, passed to `.TextPath` and
            `.FancyArrowPatch`.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        arrow_x, arrow_y : `~matplotlib.patches.FancyArrowPatch`
            Arrow x and y
        text_path_x, text_path_y : `~matplotlib.text.TextPath`
            Path for arrow labels
        p_x, p_y : `~matplotlib.patches.PathPatch`
            Patch for arrow labels
        box : `~matplotlib.offsetbox.AuxTransformBox`
            Container for the arrows and labels.

        Notes
        -----
        If *prop* is passed as a keyword argument, but *fontproperties* is
        not, then *prop* is assumed to be the intended *fontproperties*.
        Using both *prop* and *fontproperties* is not supported.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from mpl_toolkits.axes_grid1.anchored_artists import (
        ...     AnchoredDirectionArrows)
        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.random.random((10, 10)))
        >>> arrows = AnchoredDirectionArrows(ax.transAxes, '111', '110')
        >>> ax.add_artist(arrows)
        >>> fig.show()

        Using several of the optional parameters, creating downward pointing
        arrow and high contrast text labels.

        >>> import matplotlib.font_manager as fm
        >>> fontprops = fm.FontProperties(family='monospace')
        >>> arrows = AnchoredDirectionArrows(ax.transAxes, 'East', 'South',
        ...                                  loc='lower left', color='k',
        ...                                  aspect_ratio=-1, sep_x=0.02,
        ...                                  sep_y=-0.01,
        ...                                  text_props={'ec':'w', 'fc':'k'},
        ...                                  fontproperties=fontprops)
        
from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Meanline object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.violin.Meanline`
        color
            Sets the mean line color.
        visible
            Determines if a line corresponding to the sample's mean
            is shown inside the violins. If `box.visible` is turned
            on, the mean line is drawn inside the inner box.
            Otherwise, the mean line is drawn from one side of the
            violin to other.
        width
            Sets the mean line width.

        Returns
        -------
        Meanline
        
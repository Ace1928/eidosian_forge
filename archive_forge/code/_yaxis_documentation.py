from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new YAxis object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.xaxis.r
            angeslider.YAxis`
        range
            Sets the range of this axis for the rangeslider.
        rangemode
            Determines whether or not the range of this axis in the
            rangeslider use the same value than in the main plot
            when zooming in/out. If "auto", the autorange will be
            used. If "fixed", the `range` is used. If "match", the
            current range of the corresponding y-axis on the main
            subplot is used.

        Returns
        -------
        YAxis
        
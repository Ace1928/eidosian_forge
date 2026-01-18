from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Line object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.box.marker.Line`
        color
            Sets the marker.line color. It accepts either a
            specific color or an array of numbers that are mapped
            to the colorscale relative to the max and min values of
            the array or relative to `marker.line.cmin` and
            `marker.line.cmax` if set.
        outliercolor
            Sets the border line color of the outlier sample
            points. Defaults to marker.color
        outlierwidth
            Sets the border line width (in px) of the outlier
            sample points.
        width
            Sets the width (in px) of the lines bounding the marker
            points.

        Returns
        -------
        Line
        
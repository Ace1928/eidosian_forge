from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Rangeslider object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.xaxis.Rangeslider`
        autorange
            Determines whether or not the range slider range is
            computed in relation to the input data. If `range` is
            provided, then `autorange` is set to False.
        bgcolor
            Sets the background color of the range slider.
        bordercolor
            Sets the border color of the range slider.
        borderwidth
            Sets the border width of the range slider.
        range
            Sets the range of the range slider. If not set,
            defaults to the full xaxis range. If the axis `type` is
            "log", then you must take the log of your desired
            range. If the axis `type` is "date", it should be date
            strings, like date data, though Date objects and unix
            milliseconds will be accepted and converted to strings.
            If the axis `type` is "category", it should be numbers,
            using the scale where each category is assigned a
            serial number from zero in the order it appears.
        thickness
            The height of the range slider as a fraction of the
            total plot area height.
        visible
            Determines whether or not the range slider will be
            visible. If visible, perpendicular axes will be set to
            `fixedrange`
        yaxis
            :class:`plotly.graph_objects.layout.xaxis.rangeslider.Y
            Axis` instance or dict with compatible properties

        Returns
        -------
        Rangeslider
        
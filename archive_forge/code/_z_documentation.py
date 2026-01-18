from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Z object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.surface.contours.Z`
        color
            Sets the color of the contour lines.
        end
            Sets the end contour level value. Must be more than
            `contours.start`
        highlight
            Determines whether or not contour lines about the z
            dimension are highlighted on hover.
        highlightcolor
            Sets the color of the highlighted contour lines.
        highlightwidth
            Sets the width of the highlighted contour lines.
        project
            :class:`plotly.graph_objects.surface.contours.z.Project
            ` instance or dict with compatible properties
        show
            Determines whether or not contour lines about the z
            dimension are drawn.
        size
            Sets the step between each contour level. Must be
            positive.
        start
            Sets the starting contour level value. Must be less
            than `contours.end`
        usecolormap
            An alternate to "color". Determines whether or not the
            contour lines are colored using the trace "colorscale".
        width
            Sets the width of the contour lines.

        Returns
        -------
        Z
        
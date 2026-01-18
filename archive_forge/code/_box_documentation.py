from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Box object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.violin.Box`
        fillcolor
            Sets the inner box plot fill color.
        line
            :class:`plotly.graph_objects.violin.box.Line` instance
            or dict with compatible properties
        visible
            Determines if an miniature box plot is drawn inside the
            violins.
        width
            Sets the width of the inner box plots relative to the
            violins' width. For example, with 1, the inner box
            plots are as wide as the violins.

        Returns
        -------
        Box
        
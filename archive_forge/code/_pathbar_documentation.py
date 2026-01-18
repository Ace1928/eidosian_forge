from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Pathbar object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.icicle.Pathbar`
        edgeshape
            Determines which shape is used for edges between
            `barpath` labels.
        side
            Determines on which side of the the treemap the
            `pathbar` should be presented.
        textfont
            Sets the font used inside `pathbar`.
        thickness
            Sets the thickness of `pathbar` (in px). If not
            specified the `pathbar.textfont.size` is used with 3
            pixles extra padding on each side.
        visible
            Determines if the path bar is drawn i.e. outside the
            trace `domain` and with one pixel gap.

        Returns
        -------
        Pathbar
        
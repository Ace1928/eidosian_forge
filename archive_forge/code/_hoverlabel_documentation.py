from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Hoverlabel object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.scene.a
            nnotation.Hoverlabel`
        bgcolor
            Sets the background color of the hover label. By
            default uses the annotation's `bgcolor` made opaque, or
            white if it was transparent.
        bordercolor
            Sets the border color of the hover label. By default
            uses either dark grey or white, for maximum contrast
            with `hoverlabel.bgcolor`.
        font
            Sets the hover label text font. By default uses the
            global hover font and size, with color from
            `hoverlabel.bordercolor`.

        Returns
        -------
        Hoverlabel
        
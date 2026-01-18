from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Title object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.surface.colorbar.Title`
        font
            Sets this color bar's title font. Note that the title's
            font used to be set by the now deprecated `titlefont`
            attribute.
        side
            Determines the location of color bar's title with
            respect to the color bar. Defaults to "top" when
            `orientation` if "v" and  defaults to "right" when
            `orientation` if "h". Note that the title's location
            used to be set by the now deprecated `titleside`
            attribute.
        text
            Sets the title of the color bar. Note that before the
            existence of `title.text`, the title's contents used to
            be defined as the `title` attribute itself. This
            behavior has been deprecated.

        Returns
        -------
        Title
        
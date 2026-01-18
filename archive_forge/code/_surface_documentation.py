from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Surface object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.volume.Surface`
        count
            Sets the number of iso-surfaces between minimum and
            maximum iso-values. By default this value is 2 meaning
            that only minimum and maximum surfaces would be drawn.
        fill
            Sets the fill ratio of the iso-surface. The default
            fill value of the surface is 1 meaning that they are
            entirely shaded. On the other hand Applying a `fill`
            ratio less than one would allow the creation of
            openings parallel to the edges.
        pattern
            Sets the surface pattern of the iso-surface 3-D
            sections. The default pattern of the surface is `all`
            meaning that the rest of surface elements would be
            shaded. The check options (either 1 or 2) could be used
            to draw half of the squares on the surface. Using
            various combinations of capital `A`, `B`, `C`, `D` and
            `E` may also be used to reduce the number of triangles
            on the iso-surfaces and creating other patterns of
            interest.
        show
            Hides/displays surfaces between minimum and maximum
            iso-values.

        Returns
        -------
        Surface
        
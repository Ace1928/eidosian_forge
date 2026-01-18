from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Autorangeoptions object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.scene.y
            axis.Autorangeoptions`
        clipmax
            Clip autorange maximum if it goes beyond this value.
            Has no effect when `autorangeoptions.maxallowed` is
            provided.
        clipmin
            Clip autorange minimum if it goes beyond this value.
            Has no effect when `autorangeoptions.minallowed` is
            provided.
        include
            Ensure this value is included in autorange.
        includesrc
            Sets the source reference on Chart Studio Cloud for
            `include`.
        maxallowed
            Use this value exactly as autorange maximum.
        minallowed
            Use this value exactly as autorange minimum.

        Returns
        -------
        Autorangeoptions
        
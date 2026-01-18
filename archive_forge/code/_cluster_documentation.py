from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Cluster object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.scattermapbox.Cluster`
        color
            Sets the color for each cluster step.
        colorsrc
            Sets the source reference on Chart Studio Cloud for
            `color`.
        enabled
            Determines whether clustering is enabled or disabled.
        maxzoom
            Sets the maximum zoom level. At zoom levels equal to or
            greater than this, points will never be clustered.
        opacity
            Sets the marker opacity.
        opacitysrc
            Sets the source reference on Chart Studio Cloud for
            `opacity`.
        size
            Sets the size for each cluster step.
        sizesrc
            Sets the source reference on Chart Studio Cloud for
            `size`.
        step
            Sets how many points it takes to create a cluster or
            advance to the next cluster step. Use this in
            conjunction with arrays for `size` and / or `color`. If
            an integer, steps start at multiples of this number. If
            an array, each step extends from the given value until
            one less than the next value.
        stepsrc
            Sets the source reference on Chart Studio Cloud for
            `step`.

        Returns
        -------
        Cluster
        
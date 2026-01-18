from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Polar object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.Polar`
        angularaxis
            :class:`plotly.graph_objects.layout.polar.AngularAxis`
            instance or dict with compatible properties
        bargap
            Sets the gap between bars of adjacent location
            coordinates. Values are unitless, they represent
            fractions of the minimum difference in bar positions in
            the data.
        barmode
            Determines how bars at the same location coordinate are
            displayed on the graph. With "stack", the bars are
            stacked on top of one another With "overlay", the bars
            are plotted over one another, you might need to reduce
            "opacity" to see multiple bars.
        bgcolor
            Set the background color of the subplot
        domain
            :class:`plotly.graph_objects.layout.polar.Domain`
            instance or dict with compatible properties
        gridshape
            Determines if the radial axis grid lines and angular
            axis line are drawn as "circular" sectors or as
            "linear" (polygon) sectors. Has an effect only when the
            angular axis has `type` "category". Note that
            `radialaxis.angle` is snapped to the angle of the
            closest vertex when `gridshape` is "circular" (so that
            radial axis scale is the same as the data scale).
        hole
            Sets the fraction of the radius to cut out of the polar
            subplot.
        radialaxis
            :class:`plotly.graph_objects.layout.polar.RadialAxis`
            instance or dict with compatible properties
        sector
            Sets angular span of this polar subplot with two angles
            (in degrees). Sector are assumed to be spanned in the
            counterclockwise direction with 0 corresponding to
            rightmost limit of the polar subplot.
        uirevision
            Controls persistence of user-driven changes in axis
            attributes, if not overridden in the individual axes.
            Defaults to `layout.uirevision`.

        Returns
        -------
        Polar
        
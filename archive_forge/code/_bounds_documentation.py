from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Bounds object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.mapbox.Bounds`
        east
            Sets the maximum longitude of the map (in degrees East)
            if `west`, `south` and `north` are declared.
        north
            Sets the maximum latitude of the map (in degrees North)
            if `east`, `west` and `south` are declared.
        south
            Sets the minimum latitude of the map (in degrees North)
            if `east`, `west` and `north` are declared.
        west
            Sets the minimum longitude of the map (in degrees East)
            if `east`, `south` and `north` are declared.

        Returns
        -------
        Bounds
        
from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Mapbox object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.Mapbox`
        accesstoken
            Sets the mapbox access token to be used for this mapbox
            map. Alternatively, the mapbox access token can be set
            in the configuration options under `mapboxAccessToken`.
            Note that accessToken are only required when `style`
            (e.g with values : basic, streets, outdoors, light,
            dark, satellite, satellite-streets ) and/or a layout
            layer references the Mapbox server.
        bearing
            Sets the bearing angle of the map in degrees counter-
            clockwise from North (mapbox.bearing).
        bounds
            :class:`plotly.graph_objects.layout.mapbox.Bounds`
            instance or dict with compatible properties
        center
            :class:`plotly.graph_objects.layout.mapbox.Center`
            instance or dict with compatible properties
        domain
            :class:`plotly.graph_objects.layout.mapbox.Domain`
            instance or dict with compatible properties
        layers
            A tuple of
            :class:`plotly.graph_objects.layout.mapbox.Layer`
            instances or dicts with compatible properties
        layerdefaults
            When used in a template (as
            layout.template.layout.mapbox.layerdefaults), sets the
            default property values to use for elements of
            layout.mapbox.layers
        pitch
            Sets the pitch angle of the map (in degrees, where 0
            means perpendicular to the surface of the map)
            (mapbox.pitch).
        style
            Defines the map layers that are rendered by default
            below the trace layers defined in `data`, which are
            themselves by default rendered below the layers defined
            in `layout.mapbox.layers`.  These layers can be defined
            either explicitly as a Mapbox Style object which can
            contain multiple layer definitions that load data from
            any public or private Tile Map Service (TMS or XYZ) or
            Web Map Service (WMS) or implicitly by using one of the
            built-in style objects which use WMSes which do not
            require any access tokens, or by using a default Mapbox
            style or custom Mapbox style URL, both of which require
            a Mapbox access token  Note that Mapbox access token
            can be set in the `accesstoken` attribute or in the
            `mapboxAccessToken` config option.  Mapbox Style
            objects are of the form described in the Mapbox GL JS
            documentation available at
            https://docs.mapbox.com/mapbox-gl-js/style-spec  The
            built-in plotly.js styles objects are: carto-
            darkmatter, carto-positron, open-street-map, stamen-
            terrain, stamen-toner, stamen-watercolor, white-bg  The
            built-in Mapbox styles are: basic, streets, outdoors,
            light, dark, satellite, satellite-streets  Mapbox style
            URLs are of the form:
            mapbox://mapbox.mapbox-<name>-<version>
        uirevision
            Controls persistence of user-driven changes in the
            view: `center`, `zoom`, `bearing`, `pitch`. Defaults to
            `layout.uirevision`.
        zoom
            Sets the zoom level of the map (mapbox.zoom).

        Returns
        -------
        Mapbox
        
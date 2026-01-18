from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
class Scattergeo(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'scattergeo'
    _valid_props = {'connectgaps', 'customdata', 'customdatasrc', 'featureidkey', 'fill', 'fillcolor', 'geo', 'geojson', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'hovertemplate', 'hovertemplatesrc', 'hovertext', 'hovertextsrc', 'ids', 'idssrc', 'lat', 'latsrc', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'line', 'locationmode', 'locations', 'locationssrc', 'lon', 'lonsrc', 'marker', 'meta', 'metasrc', 'mode', 'name', 'opacity', 'selected', 'selectedpoints', 'showlegend', 'stream', 'text', 'textfont', 'textposition', 'textpositionsrc', 'textsrc', 'texttemplate', 'texttemplatesrc', 'type', 'uid', 'uirevision', 'unselected', 'visible'}

    @property
    def connectgaps(self):
        """
        Determines whether or not gaps (i.e. {nan} or missing values)
        in the provided data arrays are connected.

        The 'connectgaps' property must be specified as a bool
        (either True, or False)

        Returns
        -------
        bool
        """
        return self['connectgaps']

    @connectgaps.setter
    def connectgaps(self, val):
        self['connectgaps'] = val

    @property
    def customdata(self):
        """
        Assigns extra data each datum. This may be useful when
        listening to hover, click and selection events. Note that,
        "scatter" traces also appends customdata items in the markers
        DOM elements

        The 'customdata' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        self['customdata'] = val

    @property
    def customdatasrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `customdata`.

        The 'customdatasrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        self['customdatasrc'] = val

    @property
    def featureidkey(self):
        """
        Sets the key in GeoJSON features which is used as id to match
        the items included in the `locations` array. Only has an effect
        when `geojson` is set. Support nested property, for example
        "properties.name".

        The 'featureidkey' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['featureidkey']

    @featureidkey.setter
    def featureidkey(self, val):
        self['featureidkey'] = val

    @property
    def fill(self):
        """
        Sets the area to fill with a solid color. Use with `fillcolor`
        if not "none". "toself" connects the endpoints of the trace (or
        each segment of the trace if it has gaps) into a closed shape.

        The 'fill' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                ['none', 'toself']

        Returns
        -------
        Any
        """
        return self['fill']

    @fill.setter
    def fill(self, val):
        self['fill'] = val

    @property
    def fillcolor(self):
        """
        Sets the fill color. Defaults to a half-transparent variant of
        the line color, marker color, or marker line color, whichever
        is available.

        The 'fillcolor' property is a color and may be specified as:
          - A hex string (e.g. '#ff0000')
          - An rgb/rgba string (e.g. 'rgb(255,0,0)')
          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
          - A named CSS color:
                aliceblue, antiquewhite, aqua, aquamarine, azure,
                beige, bisque, black, blanchedalmond, blue,
                blueviolet, brown, burlywood, cadetblue,
                chartreuse, chocolate, coral, cornflowerblue,
                cornsilk, crimson, cyan, darkblue, darkcyan,
                darkgoldenrod, darkgray, darkgrey, darkgreen,
                darkkhaki, darkmagenta, darkolivegreen, darkorange,
                darkorchid, darkred, darksalmon, darkseagreen,
                darkslateblue, darkslategray, darkslategrey,
                darkturquoise, darkviolet, deeppink, deepskyblue,
                dimgray, dimgrey, dodgerblue, firebrick,
                floralwhite, forestgreen, fuchsia, gainsboro,
                ghostwhite, gold, goldenrod, gray, grey, green,
                greenyellow, honeydew, hotpink, indianred, indigo,
                ivory, khaki, lavender, lavenderblush, lawngreen,
                lemonchiffon, lightblue, lightcoral, lightcyan,
                lightgoldenrodyellow, lightgray, lightgrey,
                lightgreen, lightpink, lightsalmon, lightseagreen,
                lightskyblue, lightslategray, lightslategrey,
                lightsteelblue, lightyellow, lime, limegreen,
                linen, magenta, maroon, mediumaquamarine,
                mediumblue, mediumorchid, mediumpurple,
                mediumseagreen, mediumslateblue, mediumspringgreen,
                mediumturquoise, mediumvioletred, midnightblue,
                mintcream, mistyrose, moccasin, navajowhite, navy,
                oldlace, olive, olivedrab, orange, orangered,
                orchid, palegoldenrod, palegreen, paleturquoise,
                palevioletred, papayawhip, peachpuff, peru, pink,
                plum, powderblue, purple, red, rosybrown,
                royalblue, rebeccapurple, saddlebrown, salmon,
                sandybrown, seagreen, seashell, sienna, silver,
                skyblue, slateblue, slategray, slategrey, snow,
                springgreen, steelblue, tan, teal, thistle, tomato,
                turquoise, violet, wheat, white, whitesmoke,
                yellow, yellowgreen

        Returns
        -------
        str
        """
        return self['fillcolor']

    @fillcolor.setter
    def fillcolor(self, val):
        self['fillcolor'] = val

    @property
    def geo(self):
        """
        Sets a reference between this trace's geospatial coordinates
        and a geographic map. If "geo" (the default value), the
        geospatial coordinates refer to `layout.geo`. If "geo2", the
        geospatial coordinates refer to `layout.geo2`, and so on.

        The 'geo' property is an identifier of a particular
        subplot, of type 'geo', that may be specified as the string 'geo'
        optionally followed by an integer >= 1
        (e.g. 'geo', 'geo1', 'geo2', 'geo3', etc.)

        Returns
        -------
        str
        """
        return self['geo']

    @geo.setter
    def geo(self, val):
        self['geo'] = val

    @property
    def geojson(self):
        """
        Sets optional GeoJSON data associated with this trace. If not
        given, the features on the base map are used when `locations`
        is set. It can be set as a valid GeoJSON object or as a URL
        string. Note that we only accept GeoJSONs of type
        "FeatureCollection" or "Feature" with geometries of type
        "Polygon" or "MultiPolygon".

        The 'geojson' property accepts values of any type

        Returns
        -------
        Any
        """
        return self['geojson']

    @geojson.setter
    def geojson(self, val):
        self['geojson'] = val

    @property
    def hoverinfo(self):
        """
        Determines which trace information appear on hover. If `none`
        or `skip` are set, no information is displayed upon hovering.
        But, if `none` is set, click and hover events are still fired.

        The 'hoverinfo' property is a flaglist and may be specified
        as a string containing:
          - Any combination of ['lon', 'lat', 'location', 'text', 'name'] joined with '+' characters
            (e.g. 'lon+lat')
            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')
          - A list or array of the above

        Returns
        -------
        Any|numpy.ndarray
        """
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        self['hoverinfo'] = val

    @property
    def hoverinfosrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `hoverinfo`.

        The 'hoverinfosrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['hoverinfosrc']

    @hoverinfosrc.setter
    def hoverinfosrc(self, val):
        self['hoverinfosrc'] = val

    @property
    def hoverlabel(self):
        """
        The 'hoverlabel' property is an instance of Hoverlabel
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Hoverlabel`
          - A dict of string/value properties that will be passed
            to the Hoverlabel constructor

            Supported dict properties:

                align
                    Sets the horizontal alignment of the text
                    content within hover label box. Has an effect
                    only if the hover label text spans more two or
                    more lines
                alignsrc
                    Sets the source reference on Chart Studio Cloud
                    for `align`.
                bgcolor
                    Sets the background color of the hover labels
                    for this trace
                bgcolorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `bgcolor`.
                bordercolor
                    Sets the border color of the hover labels for
                    this trace.
                bordercolorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `bordercolor`.
                font
                    Sets the font used in hover labels.
                namelength
                    Sets the default length (in number of
                    characters) of the trace name in the hover
                    labels for all traces. -1 shows the whole name
                    regardless of length. 0-3 shows the first 0-3
                    characters, and an integer >3 will show the
                    whole name if it is less than that many
                    characters, but if it is longer, will truncate
                    to `namelength - 3` characters and add an
                    ellipsis.
                namelengthsrc
                    Sets the source reference on Chart Studio Cloud
                    for `namelength`.

        Returns
        -------
        plotly.graph_objs.scattergeo.Hoverlabel
        """
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        self['hoverlabel'] = val

    @property
    def hovertemplate(self):
        """
        Template string used for rendering the information that appear
        on hover box. Note that this will override `hoverinfo`.
        Variables are inserted using %{variable}, for example "y: %{y}"
        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When
        showing info for several points, "xother" will be added to
        those with different x positions from the first point. An
        underscore before or after "(x|y)other" will add a space on
        that side, only when this field is shown. Numbers are formatted
        using d3-format's syntax %{variable:d3-format}, for example
        "Price: %{y:$.2f}".
        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for
        details on the formatting syntax. Dates are formatted using
        d3-time-format's syntax %{variable|d3-time-format}, for example
        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-
        format/tree/v2.2.3#locale_format for details on the date
        formatting syntax. The variables available in `hovertemplate`
        are the ones emitted as event data described at this link
        https://plotly.com/javascript/plotlyjs-events/#event-data.
        Additionally, every attributes that can be specified per-point
        (the ones that are `arrayOk: true`) are available.  Anything
        contained in tag `<extra>` is displayed in the secondary box,
        for example "<extra>{fullData.name}</extra>". To hide the
        secondary box completely, use an empty tag `<extra></extra>`.

        The 'hovertemplate' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string
          - A tuple, list, or one-dimensional numpy array of the above

        Returns
        -------
        str|numpy.ndarray
        """
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        self['hovertemplate'] = val

    @property
    def hovertemplatesrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `hovertemplate`.

        The 'hovertemplatesrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['hovertemplatesrc']

    @hovertemplatesrc.setter
    def hovertemplatesrc(self, val):
        self['hovertemplatesrc'] = val

    @property
    def hovertext(self):
        """
        Sets hover text elements associated with each (lon,lat) pair or
        item in `locations`. If a single string, the same string
        appears over all the data points. If an array of string, the
        items are mapped in order to the this trace's (lon,lat) or
        `locations` coordinates. To be seen, trace `hoverinfo` must
        contain a "text" flag.

        The 'hovertext' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string
          - A tuple, list, or one-dimensional numpy array of the above

        Returns
        -------
        str|numpy.ndarray
        """
        return self['hovertext']

    @hovertext.setter
    def hovertext(self, val):
        self['hovertext'] = val

    @property
    def hovertextsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `hovertext`.

        The 'hovertextsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['hovertextsrc']

    @hovertextsrc.setter
    def hovertextsrc(self, val):
        self['hovertextsrc'] = val

    @property
    def ids(self):
        """
        Assigns id labels to each datum. These ids for object constancy
        of data points during animation. Should be an array of strings,
        not numbers or any other type.

        The 'ids' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['ids']

    @ids.setter
    def ids(self, val):
        self['ids'] = val

    @property
    def idssrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `ids`.

        The 'idssrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['idssrc']

    @idssrc.setter
    def idssrc(self, val):
        self['idssrc'] = val

    @property
    def lat(self):
        """
        Sets the latitude coordinates (in degrees North).

        The 'lat' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['lat']

    @lat.setter
    def lat(self, val):
        self['lat'] = val

    @property
    def latsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `lat`.

        The 'latsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['latsrc']

    @latsrc.setter
    def latsrc(self, val):
        self['latsrc'] = val

    @property
    def legend(self):
        """
        Sets the reference to a legend to show this trace in.
        References to these legends are "legend", "legend2", "legend3",
        etc. Settings for these legends are set in the layout, under
        `layout.legend`, `layout.legend2`, etc.

        The 'legend' property is an identifier of a particular
        subplot, of type 'legend', that may be specified as the string 'legend'
        optionally followed by an integer >= 1
        (e.g. 'legend', 'legend1', 'legend2', 'legend3', etc.)

        Returns
        -------
        str
        """
        return self['legend']

    @legend.setter
    def legend(self, val):
        self['legend'] = val

    @property
    def legendgroup(self):
        """
        Sets the legend group for this trace. Traces and shapes part of
        the same legend group hide/show at the same time when toggling
        legend items.

        The 'legendgroup' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['legendgroup']

    @legendgroup.setter
    def legendgroup(self, val):
        self['legendgroup'] = val

    @property
    def legendgrouptitle(self):
        """
        The 'legendgrouptitle' property is an instance of Legendgrouptitle
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Legendgrouptitle`
          - A dict of string/value properties that will be passed
            to the Legendgrouptitle constructor

            Supported dict properties:

                font
                    Sets this legend group's title font.
                text
                    Sets the title of the legend group.

        Returns
        -------
        plotly.graph_objs.scattergeo.Legendgrouptitle
        """
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        """
        Sets the legend rank for this trace. Items and groups with
        smaller ranks are presented on top/left side while with
        "reversed" `legend.traceorder` they are on bottom/right side.
        The default legendrank is 1000, so that you can use ranks less
        than 1000 to place certain items before all unranked items, and
        ranks greater than 1000 to go after all unranked items. When
        having unranked or equal rank items shapes would be displayed
        after traces i.e. according to their order in data and layout.

        The 'legendrank' property is a number and may be specified as:
          - An int or float

        Returns
        -------
        int|float
        """
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        self['legendrank'] = val

    @property
    def legendwidth(self):
        """
        Sets the width (in px or fraction) of the legend for this
        trace.

        The 'legendwidth' property is a number and may be specified as:
          - An int or float in the interval [0, inf]

        Returns
        -------
        int|float
        """
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        self['legendwidth'] = val

    @property
    def line(self):
        """
        The 'line' property is an instance of Line
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Line`
          - A dict of string/value properties that will be passed
            to the Line constructor

            Supported dict properties:

                color
                    Sets the line color.
                dash
                    Sets the dash style of lines. Set to a dash
                    type string ("solid", "dot", "dash",
                    "longdash", "dashdot", or "longdashdot") or a
                    dash length list in px (eg "5px,10px,2px,2px").
                width
                    Sets the line width (in px).

        Returns
        -------
        plotly.graph_objs.scattergeo.Line
        """
        return self['line']

    @line.setter
    def line(self, val):
        self['line'] = val

    @property
    def locationmode(self):
        """
        Determines the set of locations used to match entries in
        `locations` to regions on the map. Values "ISO-3", "USA-
        states", *country names* correspond to features on the base map
        and value "geojson-id" corresponds to features from a custom
        GeoJSON linked to the `geojson` attribute.

        The 'locationmode' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                ['ISO-3', 'USA-states', 'country names', 'geojson-id']

        Returns
        -------
        Any
        """
        return self['locationmode']

    @locationmode.setter
    def locationmode(self, val):
        self['locationmode'] = val

    @property
    def locations(self):
        """
        Sets the coordinates via location IDs or names. Coordinates
        correspond to the centroid of each location given. See
        `locationmode` for more info.

        The 'locations' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['locations']

    @locations.setter
    def locations(self, val):
        self['locations'] = val

    @property
    def locationssrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `locations`.

        The 'locationssrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['locationssrc']

    @locationssrc.setter
    def locationssrc(self, val):
        self['locationssrc'] = val

    @property
    def lon(self):
        """
        Sets the longitude coordinates (in degrees East).

        The 'lon' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['lon']

    @lon.setter
    def lon(self, val):
        self['lon'] = val

    @property
    def lonsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `lon`.

        The 'lonsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['lonsrc']

    @lonsrc.setter
    def lonsrc(self, val):
        self['lonsrc'] = val

    @property
    def marker(self):
        """
        The 'marker' property is an instance of Marker
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Marker`
          - A dict of string/value properties that will be passed
            to the Marker constructor

            Supported dict properties:

                angle
                    Sets the marker angle in respect to `angleref`.
                angleref
                    Sets the reference for marker angle. With
                    "previous", angle 0 points along the line from
                    the previous point to this one. With "up",
                    angle 0 points toward the top of the screen.
                    With "north", angle 0 points north based on the
                    current map projection.
                anglesrc
                    Sets the source reference on Chart Studio Cloud
                    for `angle`.
                autocolorscale
                    Determines whether the colorscale is a default
                    palette (`autocolorscale: true`) or the palette
                    determined by `marker.colorscale`. Has an
                    effect only if in `marker.color` is set to a
                    numerical array. In case `colorscale` is
                    unspecified or `autocolorscale` is true, the
                    default palette will be chosen according to
                    whether numbers in the `color` array are all
                    positive, all negative or mixed.
                cauto
                    Determines whether or not the color domain is
                    computed with respect to the input data (here
                    in `marker.color`) or the bounds set in
                    `marker.cmin` and `marker.cmax` Has an effect
                    only if in `marker.color` is set to a numerical
                    array. Defaults to `false` when `marker.cmin`
                    and `marker.cmax` are set by the user.
                cmax
                    Sets the upper bound of the color domain. Has
                    an effect only if in `marker.color` is set to a
                    numerical array. Value should have the same
                    units as in `marker.color` and if set,
                    `marker.cmin` must be set as well.
                cmid
                    Sets the mid-point of the color domain by
                    scaling `marker.cmin` and/or `marker.cmax` to
                    be equidistant to this point. Has an effect
                    only if in `marker.color` is set to a numerical
                    array. Value should have the same units as in
                    `marker.color`. Has no effect when
                    `marker.cauto` is `false`.
                cmin
                    Sets the lower bound of the color domain. Has
                    an effect only if in `marker.color` is set to a
                    numerical array. Value should have the same
                    units as in `marker.color` and if set,
                    `marker.cmax` must be set as well.
                color
                    Sets the marker color. It accepts either a
                    specific color or an array of numbers that are
                    mapped to the colorscale relative to the max
                    and min values of the array or relative to
                    `marker.cmin` and `marker.cmax` if set.
                coloraxis
                    Sets a reference to a shared color axis.
                    References to these shared color axes are
                    "coloraxis", "coloraxis2", "coloraxis3", etc.
                    Settings for these shared color axes are set in
                    the layout, under `layout.coloraxis`,
                    `layout.coloraxis2`, etc. Note that multiple
                    color scales can be linked to the same color
                    axis.
                colorbar
                    :class:`plotly.graph_objects.scattergeo.marker.
                    ColorBar` instance or dict with compatible
                    properties
                colorscale
                    Sets the colorscale. Has an effect only if in
                    `marker.color` is set to a numerical array. The
                    colorscale must be an array containing arrays
                    mapping a normalized value to an rgb, rgba,
                    hex, hsl, hsv, or named color string. At
                    minimum, a mapping for the lowest (0) and
                    highest (1) values are required. For example,
                    `[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]`.
                    To control the bounds of the colorscale in
                    color space, use `marker.cmin` and
                    `marker.cmax`. Alternatively, `colorscale` may
                    be a palette name string of the following list:
                    Blackbody,Bluered,Blues,Cividis,Earth,Electric,
                    Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,Rd
                    Bu,Reds,Viridis,YlGnBu,YlOrRd.
                colorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `color`.
                gradient
                    :class:`plotly.graph_objects.scattergeo.marker.
                    Gradient` instance or dict with compatible
                    properties
                line
                    :class:`plotly.graph_objects.scattergeo.marker.
                    Line` instance or dict with compatible
                    properties
                opacity
                    Sets the marker opacity.
                opacitysrc
                    Sets the source reference on Chart Studio Cloud
                    for `opacity`.
                reversescale
                    Reverses the color mapping if true. Has an
                    effect only if in `marker.color` is set to a
                    numerical array. If true, `marker.cmin` will
                    correspond to the last color in the array and
                    `marker.cmax` will correspond to the first
                    color.
                showscale
                    Determines whether or not a colorbar is
                    displayed for this trace. Has an effect only if
                    in `marker.color` is set to a numerical array.
                size
                    Sets the marker size (in px).
                sizemin
                    Has an effect only if `marker.size` is set to a
                    numerical array. Sets the minimum size (in px)
                    of the rendered marker points.
                sizemode
                    Has an effect only if `marker.size` is set to a
                    numerical array. Sets the rule for which the
                    data in `size` is converted to pixels.
                sizeref
                    Has an effect only if `marker.size` is set to a
                    numerical array. Sets the scale factor used to
                    determine the rendered size of marker points.
                    Use with `sizemin` and `sizemode`.
                sizesrc
                    Sets the source reference on Chart Studio Cloud
                    for `size`.
                standoff
                    Moves the marker away from the data point in
                    the direction of `angle` (in px). This can be
                    useful for example if you have another marker
                    at this location and you want to point an
                    arrowhead marker at it.
                standoffsrc
                    Sets the source reference on Chart Studio Cloud
                    for `standoff`.
                symbol
                    Sets the marker symbol type. Adding 100 is
                    equivalent to appending "-open" to a symbol
                    name. Adding 200 is equivalent to appending
                    "-dot" to a symbol name. Adding 300 is
                    equivalent to appending "-open-dot" or "dot-
                    open" to a symbol name.
                symbolsrc
                    Sets the source reference on Chart Studio Cloud
                    for `symbol`.

        Returns
        -------
        plotly.graph_objs.scattergeo.Marker
        """
        return self['marker']

    @marker.setter
    def marker(self, val):
        self['marker'] = val

    @property
    def meta(self):
        """
        Assigns extra meta information associated with this trace that
        can be used in various text attributes. Attributes such as
        trace `name`, graph, axis and colorbar `title.text`, annotation
        `text` `rangeselector`, `updatemenues` and `sliders` `label`
        text all support `meta`. To access the trace `meta` values in
        an attribute in the same trace, simply use `%{meta[i]}` where
        `i` is the index or key of the `meta` item in question. To
        access trace `meta` in layout attributes, use
        `%{data[n[.meta[i]}` where `i` is the index or key of the
        `meta` and `n` is the trace index.

        The 'meta' property accepts values of any type

        Returns
        -------
        Any|numpy.ndarray
        """
        return self['meta']

    @meta.setter
    def meta(self, val):
        self['meta'] = val

    @property
    def metasrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `meta`.

        The 'metasrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        self['metasrc'] = val

    @property
    def mode(self):
        """
        Determines the drawing mode for this scatter trace. If the
        provided `mode` includes "text" then the `text` elements appear
        at the coordinates. Otherwise, the `text` elements appear on
        hover. If there are less than 20 points and the trace is not
        stacked then the default is "lines+markers". Otherwise,
        "lines".

        The 'mode' property is a flaglist and may be specified
        as a string containing:
          - Any combination of ['lines', 'markers', 'text'] joined with '+' characters
            (e.g. 'lines+markers')
            OR exactly one of ['none'] (e.g. 'none')

        Returns
        -------
        Any
        """
        return self['mode']

    @mode.setter
    def mode(self, val):
        self['mode'] = val

    @property
    def name(self):
        """
        Sets the trace name. The trace name appears as the legend item
        and on hover.

        The 'name' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['name']

    @name.setter
    def name(self, val):
        self['name'] = val

    @property
    def opacity(self):
        """
        Sets the opacity of the trace.

        The 'opacity' property is a number and may be specified as:
          - An int or float in the interval [0, 1]

        Returns
        -------
        int|float
        """
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        self['opacity'] = val

    @property
    def selected(self):
        """
        The 'selected' property is an instance of Selected
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Selected`
          - A dict of string/value properties that will be passed
            to the Selected constructor

            Supported dict properties:

                marker
                    :class:`plotly.graph_objects.scattergeo.selecte
                    d.Marker` instance or dict with compatible
                    properties
                textfont
                    :class:`plotly.graph_objects.scattergeo.selecte
                    d.Textfont` instance or dict with compatible
                    properties

        Returns
        -------
        plotly.graph_objs.scattergeo.Selected
        """
        return self['selected']

    @selected.setter
    def selected(self, val):
        self['selected'] = val

    @property
    def selectedpoints(self):
        """
        Array containing integer indices of selected points. Has an
        effect only for traces that support selections. Note that an
        empty array means an empty selection where the `unselected` are
        turned on for all points, whereas, any other non-array values
        means no selection all where the `selected` and `unselected`
        styles have no effect.

        The 'selectedpoints' property accepts values of any type

        Returns
        -------
        Any
        """
        return self['selectedpoints']

    @selectedpoints.setter
    def selectedpoints(self, val):
        self['selectedpoints'] = val

    @property
    def showlegend(self):
        """
        Determines whether or not an item corresponding to this trace
        is shown in the legend.

        The 'showlegend' property must be specified as a bool
        (either True, or False)

        Returns
        -------
        bool
        """
        return self['showlegend']

    @showlegend.setter
    def showlegend(self, val):
        self['showlegend'] = val

    @property
    def stream(self):
        """
        The 'stream' property is an instance of Stream
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Stream`
          - A dict of string/value properties that will be passed
            to the Stream constructor

            Supported dict properties:

                maxpoints
                    Sets the maximum number of points to keep on
                    the plots from an incoming stream. If
                    `maxpoints` is set to 50, only the newest 50
                    points will be displayed on the plot.
                token
                    The stream id number links a data trace on a
                    plot with a stream. See https://chart-
                    studio.plotly.com/settings for more details.

        Returns
        -------
        plotly.graph_objs.scattergeo.Stream
        """
        return self['stream']

    @stream.setter
    def stream(self, val):
        self['stream'] = val

    @property
    def text(self):
        """
        Sets text elements associated with each (lon,lat) pair or item
        in `locations`. If a single string, the same string appears
        over all the data points. If an array of string, the items are
        mapped in order to the this trace's (lon,lat) or `locations`
        coordinates. If trace `hoverinfo` contains a "text" flag and
        "hovertext" is not set, these elements will be seen in the
        hover labels.

        The 'text' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string
          - A tuple, list, or one-dimensional numpy array of the above

        Returns
        -------
        str|numpy.ndarray
        """
        return self['text']

    @text.setter
    def text(self, val):
        self['text'] = val

    @property
    def textfont(self):
        """
        Sets the text font.

        The 'textfont' property is an instance of Textfont
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Textfont`
          - A dict of string/value properties that will be passed
            to the Textfont constructor

            Supported dict properties:

                color

                colorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `color`.
                family
                    HTML font family - the typeface that will be
                    applied by the web browser. The web browser
                    will only be able to apply a font if it is
                    available on the system which it operates.
                    Provide multiple font families, separated by
                    commas, to indicate the preference in which to
                    apply fonts if they aren't available on the
                    system. The Chart Studio Cloud (at
                    https://chart-studio.plotly.com or on-premise)
                    generates images on a server, where only a
                    select number of fonts are installed and
                    supported. These include "Arial", "Balto",
                    "Courier New", "Droid Sans",, "Droid Serif",
                    "Droid Sans Mono", "Gravitas One", "Old
                    Standard TT", "Open Sans", "Overpass", "PT Sans
                    Narrow", "Raleway", "Times New Roman".
                familysrc
                    Sets the source reference on Chart Studio Cloud
                    for `family`.
                size

                sizesrc
                    Sets the source reference on Chart Studio Cloud
                    for `size`.

        Returns
        -------
        plotly.graph_objs.scattergeo.Textfont
        """
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        self['textfont'] = val

    @property
    def textposition(self):
        """
        Sets the positions of the `text` elements with respects to the
        (x,y) coordinates.

        The 'textposition' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                ['top left', 'top center', 'top right', 'middle left',
                'middle center', 'middle right', 'bottom left', 'bottom
                center', 'bottom right']
          - A tuple, list, or one-dimensional numpy array of the above

        Returns
        -------
        Any|numpy.ndarray
        """
        return self['textposition']

    @textposition.setter
    def textposition(self, val):
        self['textposition'] = val

    @property
    def textpositionsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `textposition`.

        The 'textpositionsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['textpositionsrc']

    @textpositionsrc.setter
    def textpositionsrc(self, val):
        self['textpositionsrc'] = val

    @property
    def textsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `text`.

        The 'textsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['textsrc']

    @textsrc.setter
    def textsrc(self, val):
        self['textsrc'] = val

    @property
    def texttemplate(self):
        """
        Template string used for rendering the information text that
        appear on points. Note that this will override `textinfo`.
        Variables are inserted using %{variable}, for example "y:
        %{y}". Numbers are formatted using d3-format's syntax
        %{variable:d3-format}, for example "Price: %{y:$.2f}".
        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for
        details on the formatting syntax. Dates are formatted using
        d3-time-format's syntax %{variable|d3-time-format}, for example
        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-
        format/tree/v2.2.3#locale_format for details on the date
        formatting syntax. Every attributes that can be specified per-
        point (the ones that are `arrayOk: true`) are available.
        Finally, the template string has access to variables `lat`,
        `lon`, `location` and `text`.

        The 'texttemplate' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string
          - A tuple, list, or one-dimensional numpy array of the above

        Returns
        -------
        str|numpy.ndarray
        """
        return self['texttemplate']

    @texttemplate.setter
    def texttemplate(self, val):
        self['texttemplate'] = val

    @property
    def texttemplatesrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `texttemplate`.

        The 'texttemplatesrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['texttemplatesrc']

    @texttemplatesrc.setter
    def texttemplatesrc(self, val):
        self['texttemplatesrc'] = val

    @property
    def uid(self):
        """
        Assign an id to this trace, Use this to provide object
        constancy between traces during animations and transitions.

        The 'uid' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['uid']

    @uid.setter
    def uid(self, val):
        self['uid'] = val

    @property
    def uirevision(self):
        """
        Controls persistence of some user-driven changes to the trace:
        `constraintrange` in `parcoords` traces, as well as some
        `editable: true` modifications such as `name` and
        `colorbar.title`. Defaults to `layout.uirevision`. Note that
        other user-driven trace attribute changes are controlled by
        `layout` attributes: `trace.visible` is controlled by
        `layout.legend.uirevision`, `selectedpoints` is controlled by
        `layout.selectionrevision`, and `colorbar.(x|y)` (accessible
        with `config: {editable: true}`) is controlled by
        `layout.editrevision`. Trace changes are tracked by `uid`,
        which only falls back on trace index if no `uid` is provided.
        So if your app can add/remove traces before the end of the
        `data` array, such that the same trace has a different index,
        you can still preserve user-driven changes if you give each
        trace a `uid` that stays with it as it moves.

        The 'uirevision' property accepts values of any type

        Returns
        -------
        Any
        """
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        self['uirevision'] = val

    @property
    def unselected(self):
        """
        The 'unselected' property is an instance of Unselected
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.scattergeo.Unselected`
          - A dict of string/value properties that will be passed
            to the Unselected constructor

            Supported dict properties:

                marker
                    :class:`plotly.graph_objects.scattergeo.unselec
                    ted.Marker` instance or dict with compatible
                    properties
                textfont
                    :class:`plotly.graph_objects.scattergeo.unselec
                    ted.Textfont` instance or dict with compatible
                    properties

        Returns
        -------
        plotly.graph_objs.scattergeo.Unselected
        """
        return self['unselected']

    @unselected.setter
    def unselected(self, val):
        self['unselected'] = val

    @property
    def visible(self):
        """
        Determines whether or not this trace is visible. If
        "legendonly", the trace is not drawn, but can appear as a
        legend item (provided that the legend itself is visible).

        The 'visible' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                [True, False, 'legendonly']

        Returns
        -------
        Any
        """
        return self['visible']

    @visible.setter
    def visible(self, val):
        self['visible'] = val

    @property
    def type(self):
        return self._props['type']

    @property
    def _prop_descriptions(self):
        return '        connectgaps\n            Determines whether or not gaps (i.e. {nan} or missing\n            values) in the provided data arrays are connected.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        featureidkey\n            Sets the key in GeoJSON features which is used as id to\n            match the items included in the `locations` array. Only\n            has an effect when `geojson` is set. Support nested\n            property, for example "properties.name".\n        fill\n            Sets the area to fill with a solid color. Use with\n            `fillcolor` if not "none". "toself" connects the\n            endpoints of the trace (or each segment of the trace if\n            it has gaps) into a closed shape.\n        fillcolor\n            Sets the fill color. Defaults to a half-transparent\n            variant of the line color, marker color, or marker line\n            color, whichever is available.\n        geo\n            Sets a reference between this trace\'s geospatial\n            coordinates and a geographic map. If "geo" (the default\n            value), the geospatial coordinates refer to\n            `layout.geo`. If "geo2", the geospatial coordinates\n            refer to `layout.geo2`, and so on.\n        geojson\n            Sets optional GeoJSON data associated with this trace.\n            If not given, the features on the base map are used\n            when `locations` is set. It can be set as a valid\n            GeoJSON object or as a URL string. Note that we only\n            accept GeoJSONs of type "FeatureCollection" or\n            "Feature" with geometries of type "Polygon" or\n            "MultiPolygon".\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.scattergeo.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Anything contained in tag `<extra>` is\n            displayed in the secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (lon,lat)\n            pair or item in `locations`. If a single string, the\n            same string appears over all the data points. If an\n            array of string, the items are mapped in order to the\n            this trace\'s (lon,lat) or `locations` coordinates. To\n            be seen, trace `hoverinfo` must contain a "text" flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        lat\n            Sets the latitude coordinates (in degrees North).\n        latsrc\n            Sets the source reference on Chart Studio Cloud for\n            `lat`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.scattergeo.Legendgrouptitl\n            e` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        line\n            :class:`plotly.graph_objects.scattergeo.Line` instance\n            or dict with compatible properties\n        locationmode\n            Determines the set of locations used to match entries\n            in `locations` to regions on the map. Values "ISO-3",\n            "USA-states", *country names* correspond to features on\n            the base map and value "geojson-id" corresponds to\n            features from a custom GeoJSON linked to the `geojson`\n            attribute.\n        locations\n            Sets the coordinates via location IDs or names.\n            Coordinates correspond to the centroid of each location\n            given. See `locationmode` for more info.\n        locationssrc\n            Sets the source reference on Chart Studio Cloud for\n            `locations`.\n        lon\n            Sets the longitude coordinates (in degrees East).\n        lonsrc\n            Sets the source reference on Chart Studio Cloud for\n            `lon`.\n        marker\n            :class:`plotly.graph_objects.scattergeo.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines the drawing mode for this scatter trace. If\n            the provided `mode` includes "text" then the `text`\n            elements appear at the coordinates. Otherwise, the\n            `text` elements appear on hover. If there are less than\n            20 points and the trace is not stacked then the default\n            is "lines+markers". Otherwise, "lines".\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the trace.\n        selected\n            :class:`plotly.graph_objects.scattergeo.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.scattergeo.Stream`\n            instance or dict with compatible properties\n        text\n            Sets text elements associated with each (lon,lat) pair\n            or item in `locations`. If a single string, the same\n            string appears over all the data points. If an array of\n            string, the items are mapped in order to the this\n            trace\'s (lon,lat) or `locations` coordinates. If trace\n            `hoverinfo` contains a "text" flag and "hovertext" is\n            not set, these elements will be seen in the hover\n            labels.\n        textfont\n            Sets the text font.\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `lat`, `lon`, `location` and `text`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.scattergeo.Unselected`\n            instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, connectgaps=None, customdata=None, customdatasrc=None, featureidkey=None, fill=None, fillcolor=None, geo=None, geojson=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, hovertemplate=None, hovertemplatesrc=None, hovertext=None, hovertextsrc=None, ids=None, idssrc=None, lat=None, latsrc=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, line=None, locationmode=None, locations=None, locationssrc=None, lon=None, lonsrc=None, marker=None, meta=None, metasrc=None, mode=None, name=None, opacity=None, selected=None, selectedpoints=None, showlegend=None, stream=None, text=None, textfont=None, textposition=None, textpositionsrc=None, textsrc=None, texttemplate=None, texttemplatesrc=None, uid=None, uirevision=None, unselected=None, visible=None, **kwargs):
        """
        Construct a new Scattergeo object

        The data visualized as scatter point or lines on a geographic
        map is provided either by longitude/latitude pairs in `lon` and
        `lat` respectively or by geographic location IDs or names in
        `locations`.

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.Scattergeo`
        connectgaps
            Determines whether or not gaps (i.e. {nan} or missing
            values) in the provided data arrays are connected.
        customdata
            Assigns extra data each datum. This may be useful when
            listening to hover, click and selection events. Note
            that, "scatter" traces also appends customdata items in
            the markers DOM elements
        customdatasrc
            Sets the source reference on Chart Studio Cloud for
            `customdata`.
        featureidkey
            Sets the key in GeoJSON features which is used as id to
            match the items included in the `locations` array. Only
            has an effect when `geojson` is set. Support nested
            property, for example "properties.name".
        fill
            Sets the area to fill with a solid color. Use with
            `fillcolor` if not "none". "toself" connects the
            endpoints of the trace (or each segment of the trace if
            it has gaps) into a closed shape.
        fillcolor
            Sets the fill color. Defaults to a half-transparent
            variant of the line color, marker color, or marker line
            color, whichever is available.
        geo
            Sets a reference between this trace's geospatial
            coordinates and a geographic map. If "geo" (the default
            value), the geospatial coordinates refer to
            `layout.geo`. If "geo2", the geospatial coordinates
            refer to `layout.geo2`, and so on.
        geojson
            Sets optional GeoJSON data associated with this trace.
            If not given, the features on the base map are used
            when `locations` is set. It can be set as a valid
            GeoJSON object or as a URL string. Note that we only
            accept GeoJSONs of type "FeatureCollection" or
            "Feature" with geometries of type "Polygon" or
            "MultiPolygon".
        hoverinfo
            Determines which trace information appear on hover. If
            `none` or `skip` are set, no information is displayed
            upon hovering. But, if `none` is set, click and hover
            events are still fired.
        hoverinfosrc
            Sets the source reference on Chart Studio Cloud for
            `hoverinfo`.
        hoverlabel
            :class:`plotly.graph_objects.scattergeo.Hoverlabel`
            instance or dict with compatible properties
        hovertemplate
            Template string used for rendering the information that
            appear on hover box. Note that this will override
            `hoverinfo`. Variables are inserted using %{variable},
            for example "y: %{y}" as well as %{xother}, {%_xother},
            {%_xother_}, {%xother_}. When showing info for several
            points, "xother" will be added to those with different
            x positions from the first point. An underscore before
            or after "(x|y)other" will add a space on that side,
            only when this field is shown. Numbers are formatted
            using d3-format's syntax %{variable:d3-format}, for
            example "Price: %{y:$.2f}".
            https://github.com/d3/d3-format/tree/v1.4.5#d3-format
            for details on the formatting syntax. Dates are
            formatted using d3-time-format's syntax
            %{variable|d3-time-format}, for example "Day:
            %{2019-01-01|%A}". https://github.com/d3/d3-time-
            format/tree/v2.2.3#locale_format for details on the
            date formatting syntax. The variables available in
            `hovertemplate` are the ones emitted as event data
            described at this link
            https://plotly.com/javascript/plotlyjs-events/#event-
            data. Additionally, every attributes that can be
            specified per-point (the ones that are `arrayOk: true`)
            are available.  Anything contained in tag `<extra>` is
            displayed in the secondary box, for example
            "<extra>{fullData.name}</extra>". To hide the secondary
            box completely, use an empty tag `<extra></extra>`.
        hovertemplatesrc
            Sets the source reference on Chart Studio Cloud for
            `hovertemplate`.
        hovertext
            Sets hover text elements associated with each (lon,lat)
            pair or item in `locations`. If a single string, the
            same string appears over all the data points. If an
            array of string, the items are mapped in order to the
            this trace's (lon,lat) or `locations` coordinates. To
            be seen, trace `hoverinfo` must contain a "text" flag.
        hovertextsrc
            Sets the source reference on Chart Studio Cloud for
            `hovertext`.
        ids
            Assigns id labels to each datum. These ids for object
            constancy of data points during animation. Should be an
            array of strings, not numbers or any other type.
        idssrc
            Sets the source reference on Chart Studio Cloud for
            `ids`.
        lat
            Sets the latitude coordinates (in degrees North).
        latsrc
            Sets the source reference on Chart Studio Cloud for
            `lat`.
        legend
            Sets the reference to a legend to show this trace in.
            References to these legends are "legend", "legend2",
            "legend3", etc. Settings for these legends are set in
            the layout, under `layout.legend`, `layout.legend2`,
            etc.
        legendgroup
            Sets the legend group for this trace. Traces and shapes
            part of the same legend group hide/show at the same
            time when toggling legend items.
        legendgrouptitle
            :class:`plotly.graph_objects.scattergeo.Legendgrouptitl
            e` instance or dict with compatible properties
        legendrank
            Sets the legend rank for this trace. Items and groups
            with smaller ranks are presented on top/left side while
            with "reversed" `legend.traceorder` they are on
            bottom/right side. The default legendrank is 1000, so
            that you can use ranks less than 1000 to place certain
            items before all unranked items, and ranks greater than
            1000 to go after all unranked items. When having
            unranked or equal rank items shapes would be displayed
            after traces i.e. according to their order in data and
            layout.
        legendwidth
            Sets the width (in px or fraction) of the legend for
            this trace.
        line
            :class:`plotly.graph_objects.scattergeo.Line` instance
            or dict with compatible properties
        locationmode
            Determines the set of locations used to match entries
            in `locations` to regions on the map. Values "ISO-3",
            "USA-states", *country names* correspond to features on
            the base map and value "geojson-id" corresponds to
            features from a custom GeoJSON linked to the `geojson`
            attribute.
        locations
            Sets the coordinates via location IDs or names.
            Coordinates correspond to the centroid of each location
            given. See `locationmode` for more info.
        locationssrc
            Sets the source reference on Chart Studio Cloud for
            `locations`.
        lon
            Sets the longitude coordinates (in degrees East).
        lonsrc
            Sets the source reference on Chart Studio Cloud for
            `lon`.
        marker
            :class:`plotly.graph_objects.scattergeo.Marker`
            instance or dict with compatible properties
        meta
            Assigns extra meta information associated with this
            trace that can be used in various text attributes.
            Attributes such as trace `name`, graph, axis and
            colorbar `title.text`, annotation `text`
            `rangeselector`, `updatemenues` and `sliders` `label`
            text all support `meta`. To access the trace `meta`
            values in an attribute in the same trace, simply use
            `%{meta[i]}` where `i` is the index or key of the
            `meta` item in question. To access trace `meta` in
            layout attributes, use `%{data[n[.meta[i]}` where `i`
            is the index or key of the `meta` and `n` is the trace
            index.
        metasrc
            Sets the source reference on Chart Studio Cloud for
            `meta`.
        mode
            Determines the drawing mode for this scatter trace. If
            the provided `mode` includes "text" then the `text`
            elements appear at the coordinates. Otherwise, the
            `text` elements appear on hover. If there are less than
            20 points and the trace is not stacked then the default
            is "lines+markers". Otherwise, "lines".
        name
            Sets the trace name. The trace name appears as the
            legend item and on hover.
        opacity
            Sets the opacity of the trace.
        selected
            :class:`plotly.graph_objects.scattergeo.Selected`
            instance or dict with compatible properties
        selectedpoints
            Array containing integer indices of selected points.
            Has an effect only for traces that support selections.
            Note that an empty array means an empty selection where
            the `unselected` are turned on for all points, whereas,
            any other non-array values means no selection all where
            the `selected` and `unselected` styles have no effect.
        showlegend
            Determines whether or not an item corresponding to this
            trace is shown in the legend.
        stream
            :class:`plotly.graph_objects.scattergeo.Stream`
            instance or dict with compatible properties
        text
            Sets text elements associated with each (lon,lat) pair
            or item in `locations`. If a single string, the same
            string appears over all the data points. If an array of
            string, the items are mapped in order to the this
            trace's (lon,lat) or `locations` coordinates. If trace
            `hoverinfo` contains a "text" flag and "hovertext" is
            not set, these elements will be seen in the hover
            labels.
        textfont
            Sets the text font.
        textposition
            Sets the positions of the `text` elements with respects
            to the (x,y) coordinates.
        textpositionsrc
            Sets the source reference on Chart Studio Cloud for
            `textposition`.
        textsrc
            Sets the source reference on Chart Studio Cloud for
            `text`.
        texttemplate
            Template string used for rendering the information text
            that appear on points. Note that this will override
            `textinfo`. Variables are inserted using %{variable},
            for example "y: %{y}". Numbers are formatted using
            d3-format's syntax %{variable:d3-format}, for example
            "Price: %{y:$.2f}".
            https://github.com/d3/d3-format/tree/v1.4.5#d3-format
            for details on the formatting syntax. Dates are
            formatted using d3-time-format's syntax
            %{variable|d3-time-format}, for example "Day:
            %{2019-01-01|%A}". https://github.com/d3/d3-time-
            format/tree/v2.2.3#locale_format for details on the
            date formatting syntax. Every attributes that can be
            specified per-point (the ones that are `arrayOk: true`)
            are available. Finally, the template string has access
            to variables `lat`, `lon`, `location` and `text`.
        texttemplatesrc
            Sets the source reference on Chart Studio Cloud for
            `texttemplate`.
        uid
            Assign an id to this trace, Use this to provide object
            constancy between traces during animations and
            transitions.
        uirevision
            Controls persistence of some user-driven changes to the
            trace: `constraintrange` in `parcoords` traces, as well
            as some `editable: true` modifications such as `name`
            and `colorbar.title`. Defaults to `layout.uirevision`.
            Note that other user-driven trace attribute changes are
            controlled by `layout` attributes: `trace.visible` is
            controlled by `layout.legend.uirevision`,
            `selectedpoints` is controlled by
            `layout.selectionrevision`, and `colorbar.(x|y)`
            (accessible with `config: {editable: true}`) is
            controlled by `layout.editrevision`. Trace changes are
            tracked by `uid`, which only falls back on trace index
            if no `uid` is provided. So if your app can add/remove
            traces before the end of the `data` array, such that
            the same trace has a different index, you can still
            preserve user-driven changes if you give each trace a
            `uid` that stays with it as it moves.
        unselected
            :class:`plotly.graph_objects.scattergeo.Unselected`
            instance or dict with compatible properties
        visible
            Determines whether or not this trace is visible. If
            "legendonly", the trace is not drawn, but can appear as
            a legend item (provided that the legend itself is
            visible).

        Returns
        -------
        Scattergeo
        """
        super(Scattergeo, self).__init__('scattergeo')
        if '_parent' in kwargs:
            self._parent = kwargs['_parent']
            return
        if arg is None:
            arg = {}
        elif isinstance(arg, self.__class__):
            arg = arg.to_plotly_json()
        elif isinstance(arg, dict):
            arg = _copy.copy(arg)
        else:
            raise ValueError('The first argument to the plotly.graph_objs.Scattergeo\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Scattergeo`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('connectgaps', None)
        _v = connectgaps if connectgaps is not None else _v
        if _v is not None:
            self['connectgaps'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
        _v = arg.pop('featureidkey', None)
        _v = featureidkey if featureidkey is not None else _v
        if _v is not None:
            self['featureidkey'] = _v
        _v = arg.pop('fill', None)
        _v = fill if fill is not None else _v
        if _v is not None:
            self['fill'] = _v
        _v = arg.pop('fillcolor', None)
        _v = fillcolor if fillcolor is not None else _v
        if _v is not None:
            self['fillcolor'] = _v
        _v = arg.pop('geo', None)
        _v = geo if geo is not None else _v
        if _v is not None:
            self['geo'] = _v
        _v = arg.pop('geojson', None)
        _v = geojson if geojson is not None else _v
        if _v is not None:
            self['geojson'] = _v
        _v = arg.pop('hoverinfo', None)
        _v = hoverinfo if hoverinfo is not None else _v
        if _v is not None:
            self['hoverinfo'] = _v
        _v = arg.pop('hoverinfosrc', None)
        _v = hoverinfosrc if hoverinfosrc is not None else _v
        if _v is not None:
            self['hoverinfosrc'] = _v
        _v = arg.pop('hoverlabel', None)
        _v = hoverlabel if hoverlabel is not None else _v
        if _v is not None:
            self['hoverlabel'] = _v
        _v = arg.pop('hovertemplate', None)
        _v = hovertemplate if hovertemplate is not None else _v
        if _v is not None:
            self['hovertemplate'] = _v
        _v = arg.pop('hovertemplatesrc', None)
        _v = hovertemplatesrc if hovertemplatesrc is not None else _v
        if _v is not None:
            self['hovertemplatesrc'] = _v
        _v = arg.pop('hovertext', None)
        _v = hovertext if hovertext is not None else _v
        if _v is not None:
            self['hovertext'] = _v
        _v = arg.pop('hovertextsrc', None)
        _v = hovertextsrc if hovertextsrc is not None else _v
        if _v is not None:
            self['hovertextsrc'] = _v
        _v = arg.pop('ids', None)
        _v = ids if ids is not None else _v
        if _v is not None:
            self['ids'] = _v
        _v = arg.pop('idssrc', None)
        _v = idssrc if idssrc is not None else _v
        if _v is not None:
            self['idssrc'] = _v
        _v = arg.pop('lat', None)
        _v = lat if lat is not None else _v
        if _v is not None:
            self['lat'] = _v
        _v = arg.pop('latsrc', None)
        _v = latsrc if latsrc is not None else _v
        if _v is not None:
            self['latsrc'] = _v
        _v = arg.pop('legend', None)
        _v = legend if legend is not None else _v
        if _v is not None:
            self['legend'] = _v
        _v = arg.pop('legendgroup', None)
        _v = legendgroup if legendgroup is not None else _v
        if _v is not None:
            self['legendgroup'] = _v
        _v = arg.pop('legendgrouptitle', None)
        _v = legendgrouptitle if legendgrouptitle is not None else _v
        if _v is not None:
            self['legendgrouptitle'] = _v
        _v = arg.pop('legendrank', None)
        _v = legendrank if legendrank is not None else _v
        if _v is not None:
            self['legendrank'] = _v
        _v = arg.pop('legendwidth', None)
        _v = legendwidth if legendwidth is not None else _v
        if _v is not None:
            self['legendwidth'] = _v
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('locationmode', None)
        _v = locationmode if locationmode is not None else _v
        if _v is not None:
            self['locationmode'] = _v
        _v = arg.pop('locations', None)
        _v = locations if locations is not None else _v
        if _v is not None:
            self['locations'] = _v
        _v = arg.pop('locationssrc', None)
        _v = locationssrc if locationssrc is not None else _v
        if _v is not None:
            self['locationssrc'] = _v
        _v = arg.pop('lon', None)
        _v = lon if lon is not None else _v
        if _v is not None:
            self['lon'] = _v
        _v = arg.pop('lonsrc', None)
        _v = lonsrc if lonsrc is not None else _v
        if _v is not None:
            self['lonsrc'] = _v
        _v = arg.pop('marker', None)
        _v = marker if marker is not None else _v
        if _v is not None:
            self['marker'] = _v
        _v = arg.pop('meta', None)
        _v = meta if meta is not None else _v
        if _v is not None:
            self['meta'] = _v
        _v = arg.pop('metasrc', None)
        _v = metasrc if metasrc is not None else _v
        if _v is not None:
            self['metasrc'] = _v
        _v = arg.pop('mode', None)
        _v = mode if mode is not None else _v
        if _v is not None:
            self['mode'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('selected', None)
        _v = selected if selected is not None else _v
        if _v is not None:
            self['selected'] = _v
        _v = arg.pop('selectedpoints', None)
        _v = selectedpoints if selectedpoints is not None else _v
        if _v is not None:
            self['selectedpoints'] = _v
        _v = arg.pop('showlegend', None)
        _v = showlegend if showlegend is not None else _v
        if _v is not None:
            self['showlegend'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('textposition', None)
        _v = textposition if textposition is not None else _v
        if _v is not None:
            self['textposition'] = _v
        _v = arg.pop('textpositionsrc', None)
        _v = textpositionsrc if textpositionsrc is not None else _v
        if _v is not None:
            self['textpositionsrc'] = _v
        _v = arg.pop('textsrc', None)
        _v = textsrc if textsrc is not None else _v
        if _v is not None:
            self['textsrc'] = _v
        _v = arg.pop('texttemplate', None)
        _v = texttemplate if texttemplate is not None else _v
        if _v is not None:
            self['texttemplate'] = _v
        _v = arg.pop('texttemplatesrc', None)
        _v = texttemplatesrc if texttemplatesrc is not None else _v
        if _v is not None:
            self['texttemplatesrc'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('unselected', None)
        _v = unselected if unselected is not None else _v
        if _v is not None:
            self['unselected'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._props['type'] = 'scattergeo'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
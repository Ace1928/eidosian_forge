from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Link object

        The links of the Sankey plot.

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.sankey.Link`
        arrowlen
            Sets the length (in px) of the links arrow, if 0 no
            arrow will be drawn.
        color
            Sets the `link` color. It can be a single value, or an
            array for specifying color for each `link`. If
            `link.color` is omitted, then by default, a translucent
            grey link will be used.
        colorscales
            A tuple of
            :class:`plotly.graph_objects.sankey.link.Colorscale`
            instances or dicts with compatible properties
        colorscaledefaults
            When used in a template (as
            layout.template.data.sankey.link.colorscaledefaults),
            sets the default property values to use for elements of
            sankey.link.colorscales
        colorsrc
            Sets the source reference on Chart Studio Cloud for
            `color`.
        customdata
            Assigns extra data to each link.
        customdatasrc
            Sets the source reference on Chart Studio Cloud for
            `customdata`.
        hovercolor
            Sets the `link` hover color. It can be a single value,
            or an array for specifying hover colors for each
            `link`. If `link.hovercolor` is omitted, then by
            default, links will become slightly more opaque when
            hovered over.
        hovercolorsrc
            Sets the source reference on Chart Studio Cloud for
            `hovercolor`.
        hoverinfo
            Determines which trace information appear when hovering
            links. If `none` or `skip` are set, no information is
            displayed upon hovering. But, if `none` is set, click
            and hover events are still fired.
        hoverlabel
            :class:`plotly.graph_objects.sankey.link.Hoverlabel`
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
            are available.  Variables `source` and `target` are
            node objects.Finally, the template string has access to
            variables `value` and `label`. Anything contained in
            tag `<extra>` is displayed in the secondary box, for
            example "<extra>{fullData.name}</extra>". To hide the
            secondary box completely, use an empty tag
            `<extra></extra>`.
        hovertemplatesrc
            Sets the source reference on Chart Studio Cloud for
            `hovertemplate`.
        label
            The shown name of the link.
        labelsrc
            Sets the source reference on Chart Studio Cloud for
            `label`.
        line
            :class:`plotly.graph_objects.sankey.link.Line` instance
            or dict with compatible properties
        source
            An integer number `[0..nodes.length - 1]` that
            represents the source node.
        sourcesrc
            Sets the source reference on Chart Studio Cloud for
            `source`.
        target
            An integer number `[0..nodes.length - 1]` that
            represents the target node.
        targetsrc
            Sets the source reference on Chart Studio Cloud for
            `target`.
        value
            A numeric value representing the flow volume value.
        valuesrc
            Sets the source reference on Chart Studio Cloud for
            `value`.

        Returns
        -------
        Link
        
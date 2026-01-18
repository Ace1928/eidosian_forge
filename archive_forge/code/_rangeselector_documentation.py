from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Rangeselector object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.xaxis.Rangeselector`
        activecolor
            Sets the background color of the active range selector
            button.
        bgcolor
            Sets the background color of the range selector
            buttons.
        bordercolor
            Sets the color of the border enclosing the range
            selector.
        borderwidth
            Sets the width (in px) of the border enclosing the
            range selector.
        buttons
            Sets the specifications for each buttons. By default, a
            range selector comes with no buttons.
        buttondefaults
            When used in a template (as layout.template.layout.xaxi
            s.rangeselector.buttondefaults), sets the default
            property values to use for elements of
            layout.xaxis.rangeselector.buttons
        font
            Sets the font of the range selector button text.
        visible
            Determines whether or not this range selector is
            visible. Note that range selectors are only available
            for x axes of `type` set to or auto-typed to "date".
        x
            Sets the x position (in normalized coordinates) of the
            range selector.
        xanchor
            Sets the range selector's horizontal position anchor.
            This anchor binds the `x` position to the "left",
            "center" or "right" of the range selector.
        y
            Sets the y position (in normalized coordinates) of the
            range selector.
        yanchor
            Sets the range selector's vertical position anchor This
            anchor binds the `y` position to the "top", "middle" or
            "bottom" of the range selector.

        Returns
        -------
        Rangeselector
        
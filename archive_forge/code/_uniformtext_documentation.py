from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Uniformtext object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.Uniformtext`
        minsize
            Sets the minimum text size between traces of the same
            type.
        mode
            Determines how the font size for various text elements
            are uniformed between each trace type. If the computed
            text sizes were smaller than the minimum size defined
            by `uniformtext.minsize` using "hide" option hides the
            text; and using "show" option shows the text without
            further downscaling. Please note that if the size
            defined by `minsize` is greater than the font size
            defined by trace, then the `minsize` is used.

        Returns
        -------
        Uniformtext
        
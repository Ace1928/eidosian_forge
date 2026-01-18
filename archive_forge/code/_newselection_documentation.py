from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Newselection object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.Newselection`
        line
            :class:`plotly.graph_objects.layout.newselection.Line`
            instance or dict with compatible properties
        mode
            Describes how a new selection is created. If
            `immediate`, a new selection is created after first
            mouse up. If `gradual`, a new selection is not created
            after first mouse. By adding to and subtracting from
            the initial selection, this option allows declaring
            extra outlines of the selection.

        Returns
        -------
        Newselection
        
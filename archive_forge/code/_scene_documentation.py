from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Scene object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.Scene`
        annotations
            A tuple of
            :class:`plotly.graph_objects.layout.scene.Annotation`
            instances or dicts with compatible properties
        annotationdefaults
            When used in a template (as
            layout.template.layout.scene.annotationdefaults), sets
            the default property values to use for elements of
            layout.scene.annotations
        aspectmode
            If "cube", this scene's axes are drawn as a cube,
            regardless of the axes' ranges. If "data", this scene's
            axes are drawn in proportion with the axes' ranges. If
            "manual", this scene's axes are drawn in proportion
            with the input of "aspectratio" (the default behavior
            if "aspectratio" is provided). If "auto", this scene's
            axes are drawn using the results of "data" except when
            one axis is more than four times the size of the two
            others, where in that case the results of "cube" are
            used.
        aspectratio
            Sets this scene's axis aspectratio.
        bgcolor

        camera
            :class:`plotly.graph_objects.layout.scene.Camera`
            instance or dict with compatible properties
        domain
            :class:`plotly.graph_objects.layout.scene.Domain`
            instance or dict with compatible properties
        dragmode
            Determines the mode of drag interactions for this
            scene.
        hovermode
            Determines the mode of hover interactions for this
            scene.
        uirevision
            Controls persistence of user-driven changes in camera
            attributes. Defaults to `layout.uirevision`.
        xaxis
            :class:`plotly.graph_objects.layout.scene.XAxis`
            instance or dict with compatible properties
        yaxis
            :class:`plotly.graph_objects.layout.scene.YAxis`
            instance or dict with compatible properties
        zaxis
            :class:`plotly.graph_objects.layout.scene.ZAxis`
            instance or dict with compatible properties

        Returns
        -------
        Scene
        
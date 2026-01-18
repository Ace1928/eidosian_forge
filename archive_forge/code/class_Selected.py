from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
class Selected(_BaseTraceHierarchyType):
    _parent_path_str = 'box'
    _path_str = 'box.selected'
    _valid_props = {'marker'}

    @property
    def marker(self):
        """
        The 'marker' property is an instance of Marker
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.box.selected.Marker`
          - A dict of string/value properties that will be passed
            to the Marker constructor

            Supported dict properties:

                color
                    Sets the marker color of selected points.
                opacity
                    Sets the marker opacity of selected points.
                size
                    Sets the marker size of selected points.

        Returns
        -------
        plotly.graph_objs.box.selected.Marker
        """
        return self['marker']

    @marker.setter
    def marker(self, val):
        self['marker'] = val

    @property
    def _prop_descriptions(self):
        return '        marker\n            :class:`plotly.graph_objects.box.selected.Marker`\n            instance or dict with compatible properties\n        '

    def __init__(self, arg=None, marker=None, **kwargs):
        """
        Construct a new Selected object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.box.Selected`
        marker
            :class:`plotly.graph_objects.box.selected.Marker`
            instance or dict with compatible properties

        Returns
        -------
        Selected
        """
        super(Selected, self).__init__('selected')
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
            raise ValueError('The first argument to the plotly.graph_objs.box.Selected\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.box.Selected`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('marker', None)
        _v = marker if marker is not None else _v
        if _v is not None:
            self['marker'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
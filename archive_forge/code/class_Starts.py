from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
class Starts(_BaseTraceHierarchyType):
    _parent_path_str = 'streamtube'
    _path_str = 'streamtube.starts'
    _valid_props = {'x', 'xsrc', 'y', 'ysrc', 'z', 'zsrc'}

    @property
    def x(self):
        """
        Sets the x components of the starting position of the
        streamtubes

        The 'x' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['x']

    @x.setter
    def x(self, val):
        self['x'] = val

    @property
    def xsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `x`.

        The 'xsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['xsrc']

    @xsrc.setter
    def xsrc(self, val):
        self['xsrc'] = val

    @property
    def y(self):
        """
        Sets the y components of the starting position of the
        streamtubes

        The 'y' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['y']

    @y.setter
    def y(self, val):
        self['y'] = val

    @property
    def ysrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `y`.

        The 'ysrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['ysrc']

    @ysrc.setter
    def ysrc(self, val):
        self['ysrc'] = val

    @property
    def z(self):
        """
        Sets the z components of the starting position of the
        streamtubes

        The 'z' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['z']

    @z.setter
    def z(self, val):
        self['z'] = val

    @property
    def zsrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `z`.

        The 'zsrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['zsrc']

    @zsrc.setter
    def zsrc(self, val):
        self['zsrc'] = val

    @property
    def _prop_descriptions(self):
        return '        x\n            Sets the x components of the starting position of the\n            streamtubes\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y components of the starting position of the\n            streamtubes\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n        z\n            Sets the z components of the starting position of the\n            streamtubes\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n        '

    def __init__(self, arg=None, x=None, xsrc=None, y=None, ysrc=None, z=None, zsrc=None, **kwargs):
        """
        Construct a new Starts object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.streamtube.Starts`
        x
            Sets the x components of the starting position of the
            streamtubes
        xsrc
            Sets the source reference on Chart Studio Cloud for
            `x`.
        y
            Sets the y components of the starting position of the
            streamtubes
        ysrc
            Sets the source reference on Chart Studio Cloud for
            `y`.
        z
            Sets the z components of the starting position of the
            streamtubes
        zsrc
            Sets the source reference on Chart Studio Cloud for
            `z`.

        Returns
        -------
        Starts
        """
        super(Starts, self).__init__('starts')
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
            raise ValueError('The first argument to the plotly.graph_objs.streamtube.Starts\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.streamtube.Starts`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('xsrc', None)
        _v = xsrc if xsrc is not None else _v
        if _v is not None:
            self['xsrc'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('ysrc', None)
        _v = ysrc if ysrc is not None else _v
        if _v is not None:
            self['ysrc'] = _v
        _v = arg.pop('z', None)
        _v = z if z is not None else _v
        if _v is not None:
            self['z'] = _v
        _v = arg.pop('zsrc', None)
        _v = zsrc if zsrc is not None else _v
        if _v is not None:
            self['zsrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
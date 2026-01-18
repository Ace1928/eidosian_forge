from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
class Title(_BaseTraceHierarchyType):
    _parent_path_str = 'surface.colorbar'
    _path_str = 'surface.colorbar.title'
    _valid_props = {'font', 'side', 'text'}

    @property
    def font(self):
        """
        Sets this color bar's title font. Note that the title's font
        used to be set by the now deprecated `titlefont` attribute.

        The 'font' property is an instance of Font
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.surface.colorbar.title.Font`
          - A dict of string/value properties that will be passed
            to the Font constructor

            Supported dict properties:

                color

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
                size

        Returns
        -------
        plotly.graph_objs.surface.colorbar.title.Font
        """
        return self['font']

    @font.setter
    def font(self, val):
        self['font'] = val

    @property
    def side(self):
        """
        Determines the location of color bar's title with respect to
        the color bar. Defaults to "top" when `orientation` if "v" and
        defaults to "right" when `orientation` if "h". Note that the
        title's location used to be set by the now deprecated
        `titleside` attribute.

        The 'side' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                ['right', 'top', 'bottom']

        Returns
        -------
        Any
        """
        return self['side']

    @side.setter
    def side(self, val):
        self['side'] = val

    @property
    def text(self):
        """
        Sets the title of the color bar. Note that before the existence
        of `title.text`, the title's contents used to be defined as the
        `title` attribute itself. This behavior has been deprecated.

        The 'text' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['text']

    @text.setter
    def text(self, val):
        self['text'] = val

    @property
    def _prop_descriptions(self):
        return '        font\n            Sets this color bar\'s title font. Note that the title\'s\n            font used to be set by the now deprecated `titlefont`\n            attribute.\n        side\n            Determines the location of color bar\'s title with\n            respect to the color bar. Defaults to "top" when\n            `orientation` if "v" and  defaults to "right" when\n            `orientation` if "h". Note that the title\'s location\n            used to be set by the now deprecated `titleside`\n            attribute.\n        text\n            Sets the title of the color bar. Note that before the\n            existence of `title.text`, the title\'s contents used to\n            be defined as the `title` attribute itself. This\n            behavior has been deprecated.\n        '

    def __init__(self, arg=None, font=None, side=None, text=None, **kwargs):
        """
        Construct a new Title object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.surface.colorbar.Title`
        font
            Sets this color bar's title font. Note that the title's
            font used to be set by the now deprecated `titlefont`
            attribute.
        side
            Determines the location of color bar's title with
            respect to the color bar. Defaults to "top" when
            `orientation` if "v" and  defaults to "right" when
            `orientation` if "h". Note that the title's location
            used to be set by the now deprecated `titleside`
            attribute.
        text
            Sets the title of the color bar. Note that before the
            existence of `title.text`, the title's contents used to
            be defined as the `title` attribute itself. This
            behavior has been deprecated.

        Returns
        -------
        Title
        """
        super(Title, self).__init__('title')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.colorbar.Title\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.colorbar.Title`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('side', None)
        _v = side if side is not None else _v
        if _v is not None:
            self['side'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
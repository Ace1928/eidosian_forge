from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class themeable(metaclass=RegistryHierarchyMeta):
    """
    Abstract class of things that can be themed.

    Every subclass of themeable is stored in a dict at
    [](`~plotnine.theme.themeables.themeable.register`) with the name
    of the subclass as the key.

    It is the base of a class hierarchy that uses inheritance in a
    non-traditional manner. In the textbook use of class inheritance,
    superclasses are general and subclasses are specializations. In some
    since the hierarchy used here is the opposite in that superclasses
    are more specific than subclasses.

    It is probably better to think if this hierarchy of leveraging
    Python's multiple inheritance to implement composition. For example
    the `axis_title` themeable is *composed of* the `x_axis_title` and the
    `y_axis_title`. We are just using multiple inheritance to specify
    this composition.

    When implementing a new themeable based on the ggplot2 documentation,
    it is important to keep this in mind and reverse the order of the
    "inherits from" in the documentation.

    For example, to implement,

    - `axis_title_x` - `x` axis label (element_text;
      inherits from `axis_title`)
    - `axis_title_y` - `y` axis label (element_text;
      inherits from `axis_title`)


    You would have this implementation:


    ```python
    class axis_title_x(themeable):
        ...

    class axis_title_y(themeable):
        ...

    class axis_title(axis_title_x, axis_title_y):
        ...
    ```

    If the superclasses fully implement the subclass, the body of the
    subclass should be "pass". Python(__mro__) will do the right thing.

    When a method does require implementation, call `super()`{.py}
    then add the themeable's implementation to the axes.

    Notes
    -----
    A user should never create instances of class
    [](`~plotnine.themes.themeable.Themeable`) or subclasses of it.
    """
    _omit: list[str] = []
    '\n    Properties to ignore during the apply stage.\n\n    These properties may have been used when creating the artists and\n    applying them would create a conflict or an error.\n    '

    def __init__(self, theme_element: element_base | str | float):
        self.theme_element = theme_element
        if isinstance(theme_element, element_base):
            self._properties: dict[str, Any] = theme_element.properties
        else:
            self._properties = {'value': theme_element}

    @staticmethod
    def from_class_name(name: str, theme_element: Any) -> themeable:
        """
        Create a themeable by name

        Parameters
        ----------
        name : str
            Class name
        theme_element : element object
            An element of the type required by the theme.
            For lines, text and rects it should be one of:
            [](`~plotnine.themes.element_line`),
            [](`~plotnine.themes.element_rect`),
            [](`~plotnine.themes.element_text`) or
            [](`~plotnine.themes.element_blank`)

        Returns
        -------
        out : plotnine.themes.themeable.themeable
        """
        msg = f'There no themeable element called: {name}'
        try:
            klass: Type[themeable] = themeable._registry[name]
        except KeyError as e:
            raise PlotnineError(msg) from e
        if not issubclass(klass, themeable):
            raise PlotnineError(msg)
        return klass(theme_element)

    @classmethod
    def registry(cls) -> Mapping[str, Any]:
        return themeable._registry

    def is_blank(self) -> bool:
        """
        Return True if theme_element is made of element_blank
        """
        return isinstance(self.theme_element, element_blank)

    def merge(self, other: themeable):
        """
        Merge properties of other into self

        Raises
        ------
        ValueError
            If any of the properties are blank
        """
        if self.is_blank() or other.is_blank():
            raise ValueError('Cannot merge if there is a blank.')
        else:
            self._properties.update(other._properties)

    def __eq__(self, other: object) -> bool:
        """Mostly for unittesting."""
        return other is self or (isinstance(other, type(self)) and self._properties == other._properties)

    @property
    def rcParams(self) -> dict[str, Any]:
        """
        Return themeables rcparams to an rcparam dict before plotting.

        Returns
        -------
        dict
            Dictionary of legal matplotlib parameters.

        This method should always call super(...).rcParams and
        update the dictionary that it returns with its own value, and
        return that dictionary.

        This method is called before plotting. It tends to be more
        useful for general themeables. Very specific themeables
        often cannot be be themed until they are created as a
        result of the plotting process.
        """
        return {}

    @property
    def properties(self):
        """
        Return only the properties that can be applied
        """
        d = self._properties.copy()
        for key in self._omit:
            with suppress(KeyError):
                del d[key]
        return d

    def apply(self, theme: theme):
        """
        Called by the theme to apply the themeable

        Subclasses should not have to override this method
        """
        blanks = (self.blank_figure, self.blank_ax)
        applys = (self.apply_figure, self.apply_ax)
        do_figure, do_ax = blanks if self.is_blank() else applys
        do_figure(theme.figure, theme.targets)
        for ax in theme.axs:
            do_ax(ax)

    def apply_ax(self, ax: Axes):
        """
        Called after a chart has been plotted.

        Subclasses can override this method to customize the plot
        according to the theme.

        This method should be implemented as `super().apply_ax()`{.py}
        followed by extracting the portion of the axes specific to this
        themeable then applying the properties.


        Parameters
        ----------
        ax : matplotlib.axes.Axes
        """

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        """
        Apply theme to the figure
        """

    def blank_ax(self, ax: Axes):
        """
        Blank out theme elements
        """

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        """
        Blank out elements on the figure
        """
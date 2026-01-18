from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
class WidgetContainerMixin:
    """
    Mixin class for widget containers implementing common container methods
    """

    def __getitem__(self, position) -> Widget:
        """
        Container short-cut for self.contents[position][0].base_widget
        which means "give me the child widget at position without any
        widget decorations".

        This allows for concise traversal of nested container widgets
        such as:

            my_widget[position0][position1][position2] ...
        """
        return self.contents[position][0].base_widget

    def get_focus_path(self) -> list[int | str]:
        """
        Return the .focus_position values starting from this container
        and proceeding along each child widget until reaching a leaf
        (non-container) widget.
        """
        out = []
        w = self
        while True:
            try:
                p = w.focus_position
            except IndexError:
                return out
            out.append(p)
            w = w.focus.base_widget

    def set_focus_path(self, positions: Iterable[int | str]) -> None:
        """
        Set the .focus_position property starting from this container
        widget and proceeding along newly focused child widgets.  Any
        failed assignment due do incompatible position types or invalid
        positions will raise an IndexError.

        This method may be used to restore a particular widget to the
        focus by passing in the value returned from an earlier call to
        get_focus_path().

        positions -- sequence of positions
        """
        w: Widget = self
        for p in positions:
            if p != w.focus_position:
                w.focus_position = p
            w = w.focus.base_widget

    def get_focus_widgets(self) -> list[Widget]:
        """
        Return the .focus values starting from this container
        and proceeding along each child widget until reaching a leaf
        (non-container) widget.

        Note that the list does not contain the topmost container widget
        (i.e., on which this method is called), but does include the
        lowest leaf widget.
        """
        out = []
        w = self
        while True:
            w = w.base_widget.focus
            if w is None:
                return out
            out.append(w)

    @property
    @abc.abstractmethod
    def focus(self) -> Widget:
        """
        Read-only property returning the child widget in focus for
        container widgets.  This default implementation
        always returns ``None``, indicating that this widget has no children.
        """

    def _get_focus(self) -> Widget:
        warnings.warn(f'method `{self.__class__.__name__}._get_focus` is deprecated, please use `{self.__class__.__name__}.focus` property', DeprecationWarning, stacklevel=3)
        return self.focus
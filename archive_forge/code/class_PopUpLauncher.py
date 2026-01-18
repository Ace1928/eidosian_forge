from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .overlay import Overlay
from .widget import delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
class PopUpLauncher(delegate_to_widget_mixin('_original_widget'), WidgetDecoration[WrappedWidget]):

    def __init__(self, original_widget: [WrappedWidget]) -> None:
        super().__init__(original_widget)
        self._pop_up_widget = None

    def create_pop_up(self) -> Widget:
        """
        Subclass must override this method and return a widget
        to be used for the pop-up.  This method is called once each time
        the pop-up is opened.
        """
        raise NotImplementedError('Subclass must override this method')

    def get_pop_up_parameters(self) -> PopUpParametersModel:
        """
        Subclass must override this method and have it return a dict, eg:

        {'left':0, 'top':1, 'overlay_width':30, 'overlay_height':4}

        This method is called each time this widget is rendered.
        """
        raise NotImplementedError('Subclass must override this method')

    def open_pop_up(self) -> None:
        self._pop_up_widget = self.create_pop_up()
        self._invalidate()

    def close_pop_up(self) -> None:
        self._pop_up_widget = None
        self._invalidate()

    def render(self, size, focus: bool=False) -> CompositeCanvas | Canvas:
        canv = super().render(size, focus)
        if self._pop_up_widget:
            canv = CompositeCanvas(canv)
            canv.set_pop_up(self._pop_up_widget, **self.get_pop_up_parameters())
        return canv
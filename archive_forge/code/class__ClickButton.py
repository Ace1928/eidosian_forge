from __future__ import annotations
from typing import (
import param
from bokeh.events import ButtonClick, MenuItemClick
from bokeh.models import Dropdown as _BkDropdown, Toggle as _BkToggle
from bokeh.models.ui import SVGIcon, TablerIcon
from ..io.resources import CDN_DIST
from ..links import Callback
from ..models.widgets import Button as _BkButton
from ._mixin import TooltipMixin
from .base import Widget
class _ClickButton(Widget):
    __abstract = True
    _event: ClassVar[str] = 'button_click'

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        model = super()._get_model(doc, root, parent, comm)
        self._register_events(self._event, model=model, doc=doc, comm=comm)
        return model

    def js_on_click(self, args: Dict[str, Any]={}, code: str='') -> Callback:
        """
        Allows defining a JS callback to be triggered when the button
        is clicked.

        Arguments
        ----------
        args: dict
          A mapping of objects to make available to the JS callback
        code: str
          The Javascript code to execute when the button is clicked.

        Returns
        -------
        callback: Callback
          The Callback which can be used to disable the callback.
        """
        from ..links import Callback
        return Callback(self, code={'event:' + self._event: code}, args=args)

    def jscallback(self, args: Dict[str, Any]={}, **callbacks: str) -> Callback:
        """
        Allows defining a Javascript (JS) callback to be triggered when a property
        changes on the source object. The keyword arguments define the
        properties that trigger a callback and the JS code that gets
        executed.

        Arguments
        ----------
        args: dict
          A mapping of objects to make available to the JS callback
        **callbacks: dict
          A mapping between properties on the source model and the code
          to execute when that property changes

        Returns
        -------
        callback: Callback
          The Callback which can be used to disable the callback.
        """
        for k, v in list(callbacks.items()):
            if k == 'clicks':
                k = 'event:' + self._event
            val = self._rename.get(v, v)
            if val is not None:
                callbacks[k] = val
        return Callback(self, code=callbacks, args=args)
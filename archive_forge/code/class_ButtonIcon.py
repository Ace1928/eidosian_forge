from __future__ import annotations
from typing import (
import param
from ..io.resources import CDN_DIST
from ..models import (
from ._mixin import TooltipMixin
from .base import Widget
from .button import ButtonClick, _ClickButton
class ButtonIcon(_ClickableIcon, _ClickButton, TooltipMixin):
    """
    The `ButtonIcon` widget facilitates event triggering upon button clicks.

    This widget displays a default `icon` initially. Upon being clicked, an `active_icon` appears
    for a specified `toggle_duration`.

    For instance, the `ButtonIcon` can be effectively utilized to implement a feature akin to
    ChatGPT's copy-to-clipboard button.

    The button incorporates a `value` attribute, which alternates between `False` and `True` as the
    click event is processed.

    Furthermore, it includes an `clicks` attribute, enabling subscription to click events for
    further actions or monitoring.

    Reference: https://panel.holoviz.org/reference/widgets/ButtonIcon.html

    :Example:

    >>> button_icon = pn.widgets.ButtonIcon(
    ...     icon='clipboard',
    ...     active_icon='check',
    ...     description='Copy',
    ...     toggle_duration=2000
    ... )
    """
    clicks = param.Integer(default=0, doc='\n        The number of times the button has been clicked.')
    toggle_duration = param.Integer(default=75, doc='\n        The number of milliseconds the active_icon should be shown for\n        and how long the button should be disabled for.')
    value = param.Boolean(default=False, doc='\n        Toggles from False to True while the event is being processed.')
    _widget_type = _PnButtonIcon
    _rename: ClassVar[Mapping[str, str | None]] = {**TooltipMixin._rename, 'name': 'title', 'clicks': None}
    _target_transforms: ClassVar[Mapping[str, str | None]] = {'event:button_click': None}

    def __init__(self, **params):
        click_handler = params.pop('on_click', None)
        super().__init__(**params)
        if click_handler:
            self.on_click(click_handler)

    def on_click(self, callback: Callable[[param.parameterized.Event], None]) -> param.parameterized.Watcher:
        """
        Register a callback to be executed when the button is clicked.

        The callback is given an `Event` argument declaring the number of clicks.

        Arguments
        ---------
        callback: (Callable[[param.parameterized.Event], None])
            The function to run on click events. Must accept a positional `Event` argument

        Returns
        -------
        watcher: param.Parameterized.Watcher
          A `Watcher` that executes the callback when the MenuButton is clicked.
        """
        return self.param.watch(callback, 'clicks', onlychanged=False)

    def _process_event(self, event: ButtonClick) -> None:
        """
        Process a button click event.
        """
        self.param.trigger('value')
        self.clicks += 1
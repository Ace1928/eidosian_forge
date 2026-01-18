from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import (
import param
from ..io.resources import CDN_DIST
from ..layout import Row, Tabs
from ..pane.image import ImageBase
from ..viewable import Viewable
from ..widgets.base import Widget
from ..widgets.button import Button
from ..widgets.input import FileInput, TextInput
from .feed import CallbackState, ChatFeed
from .input import ChatAreaInput
from .message import ChatMessage, _FileInputMessage
@param.depends('widgets', 'button_properties', watch=True)
def _init_widgets(self):
    """
        Initialize the input widgets.

        Returns
        -------
        The input widgets.
        """
    default_button_properties = {'send': {'icon': 'send', '_default_callback': self._click_send}, 'stop': {'icon': 'player-stop', '_default_callback': self._click_stop}, 'rerun': {'icon': 'repeat', '_default_callback': self._click_rerun}, 'undo': {'icon': 'arrow-back', '_default_callback': self._click_undo}, 'clear': {'icon': 'trash', '_default_callback': self._click_clear}}
    self._allow_revert = len(self.button_properties) == 0
    button_properties = {**default_button_properties, **self.button_properties}
    for index, (name, properties) in enumerate(button_properties.items()):
        name = name.lower()
        callback = properties.get('callback')
        post_callback = properties.get('post_callback')
        default_properties = default_button_properties.get(name) or {}
        if default_properties:
            default_callback = default_properties['_default_callback']
            callback = self._wrap_callbacks(callback=callback, post_callback=post_callback, name=name)(default_callback) if callback is not None or post_callback is not None else default_callback
        elif callback is not None and post_callback is not None:
            callback = self._wrap_callbacks(post_callback=post_callback)(callback)
        elif callback is None and post_callback is not None:
            callback = post_callback
        elif callback is None and post_callback is None:
            raise ValueError(f"A 'callback' key is required for the {name!r} button")
        icon = properties.get('icon') or default_properties.get('icon')
        self._button_data[name] = _ChatButtonData(index=index, name=name, icon=icon, objects=[], buttons=[], callback=callback)
    widgets = self.widgets
    if isinstance(self.widgets, Widget):
        widgets = [self.widgets]
    self._widgets = {}
    new_widgets = []
    for widget in widgets:
        key = widget.name or widget.__class__.__name__
        if isinstance(widget, type):
            widget = widget()
        if self._widgets.get(key) is not widget:
            self._widgets[key] = widget
            new_widgets.append(widget)
    sizing_mode = self.sizing_mode
    if sizing_mode is not None:
        if 'both' in sizing_mode or 'scale_height' in sizing_mode:
            sizing_mode = 'stretch_width'
        elif 'height' in sizing_mode:
            sizing_mode = None
    input_layout = Tabs(sizing_mode=sizing_mode, css_classes=['chat-interface-input-tabs'], stylesheets=self._stylesheets, dynamic=True)
    for name, widget in self._widgets.items():
        auto_send = isinstance(widget, tuple(self.auto_send_types)) or type(widget) in (TextInput, ChatAreaInput)
        if auto_send and widget in new_widgets:
            callback = partial(self._button_data['send'].callback, self)
            widget.param.watch(callback, 'value')
        widget.param.update(sizing_mode='stretch_width', css_classes=['chat-interface-input-widget'])
        if isinstance(widget, ChatAreaInput):
            self.link(widget, disabled='disabled_enter')
        self._buttons = {}
        for button_data in self._button_data.values():
            action = button_data.name
            try:
                visible = self.param[f'show_{action}'] if action != 'stop' else False
            except KeyError:
                visible = True
            show_expr = self.param.show_button_name.rx()
            button = Button(name=show_expr.rx.where(button_data.name.title(), ''), icon=button_data.icon, sizing_mode='stretch_width', max_width=show_expr.rx.where(90, 45), max_height=50, margin=(0, 5, 0, 0), align='center', visible=visible)
            if action != 'stop':
                self._link_disabled_loading(button)
            callback = partial(button_data.callback, self)
            button.on_click(callback)
            self._buttons[action] = button
            button_data.buttons.append(button)
        message_row = Row(widget, *list(self._buttons.values()), sizing_mode='stretch_width', css_classes=['chat-interface-input-row'], stylesheets=self._stylesheets, align='start')
        input_layout.append((name, message_row))
    if len(self._widgets) == 1:
        input_layout = input_layout[0]
    self._input_container.objects = [input_layout]
    self._input_layout = input_layout
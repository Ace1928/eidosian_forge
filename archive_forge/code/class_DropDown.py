from kivy.uix.scrollview import ScrollView
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.config import Config
class DropDown(ScrollView):
    """DropDown class. See module documentation for more information.

    :Events:
        `on_select`: data
            Fired when a selection is done. The data of the selection is passed
            in as the first argument and is what you pass in the :meth:`select`
            method as the first argument.
        `on_dismiss`:
            .. versionadded:: 1.8.0

            Fired when the DropDown is dismissed, either on selection or on
            touching outside the widget.
    """
    auto_width = BooleanProperty(True)
    'By default, the width of the dropdown will be the same as the width of\n    the attached widget. Set to False if you want to provide your own width.\n\n    :attr:`auto_width` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to True.\n    '
    max_height = NumericProperty(None, allownone=True)
    'Indicate the maximum height that the dropdown can take. If None, it will\n    take the maximum height available until the top or bottom of the screen\n    is reached.\n\n    :attr:`max_height` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to None.\n    '
    dismiss_on_select = BooleanProperty(True)
    'By default, the dropdown will be automatically dismissed when a\n    selection has been done. Set to False to prevent the dismiss.\n\n    :attr:`dismiss_on_select` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to True.\n    '
    auto_dismiss = BooleanProperty(True)
    'By default, the dropdown will be automatically dismissed when a\n    touch happens outside of it, this option allows to disable this\n    feature\n\n    :attr:`auto_dismiss` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to True.\n\n    .. versionadded:: 1.8.0\n    '
    min_state_time = NumericProperty(0)
    'Minimum time before the :class:`~kivy.uix.DropDown` is dismissed.\n    This is used to allow for the widget inside the dropdown to display\n    a down state or for the :class:`~kivy.uix.DropDown` itself to\n    display a animation for closing.\n\n    :attr:`min_state_time` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to the `Config` value `min_state_time`.\n\n    .. versionadded:: 1.10.0\n    '
    attach_to = ObjectProperty(allownone=True)
    '(internal) Property that will be set to the widget to which the\n    drop down list is attached.\n\n    The :meth:`open` method will automatically set this property whilst\n    :meth:`dismiss` will set it back to None.\n    '
    container = ObjectProperty()
    '(internal) Property that will be set to the container of the dropdown\n    list. It is a :class:`~kivy.uix.gridlayout.GridLayout` by default.\n    '
    _touch_started_inside = None
    __events__ = ('on_select', 'on_dismiss')

    def __init__(self, **kwargs):
        self._win = None
        if 'min_state_time' not in kwargs:
            self.min_state_time = float(Config.get('graphics', 'min_state_time'))
        if 'container' not in kwargs:
            c = self.container = Builder.load_string(_grid_kv)
        else:
            c = None
        if 'do_scroll_x' not in kwargs:
            self.do_scroll_x = False
        if 'size_hint' not in kwargs:
            if 'size_hint_x' not in kwargs:
                self.size_hint_x = None
            if 'size_hint_y' not in kwargs:
                self.size_hint_y = None
        super(DropDown, self).__init__(**kwargs)
        if c is not None:
            super(DropDown, self).add_widget(c)
            self.on_container(self, c)
        Window.bind(on_key_down=self.on_key_down, size=self._reposition)
        self.fbind('size', self._reposition)

    def on_key_down(self, instance, key, scancode, codepoint, modifiers):
        if key == 27 and self.get_parent_window():
            self.dismiss()
            return True

    def on_container(self, instance, value):
        if value is not None:
            self.container.bind(minimum_size=self._reposition)

    def open(self, widget):
        """Open the dropdown list and attach it to a specific widget.
        Depending on the position of the widget within the window and
        the height of the dropdown, the dropdown might be above or below
        that widget.
        """
        if self.attach_to is not None:
            self.dismiss()
        self._win = widget.get_parent_window()
        if self._win is None:
            raise DropDownException('Cannot open a dropdown list on a hidden widget')
        self.attach_to = widget
        widget.bind(pos=self._reposition, size=self._reposition)
        self._reposition()
        self._win.add_widget(self)

    def dismiss(self, *largs):
        """Remove the dropdown widget from the window and detach it from
        the attached widget.
        """
        Clock.schedule_once(self._real_dismiss, self.min_state_time)

    def _real_dismiss(self, *largs):
        if self.parent:
            self.parent.remove_widget(self)
        if self.attach_to:
            self.attach_to.unbind(pos=self._reposition, size=self._reposition)
            self.attach_to = None
        self.dispatch('on_dismiss')

    def on_dismiss(self):
        pass

    def select(self, data):
        """Call this method to trigger the `on_select` event with the `data`
        selection. The `data` can be anything you want.
        """
        self.dispatch('on_select', data)
        if self.dismiss_on_select:
            self.dismiss()

    def on_select(self, data):
        pass

    def add_widget(self, *args, **kwargs):
        if self.container:
            return self.container.add_widget(*args, **kwargs)
        return super(DropDown, self).add_widget(*args, **kwargs)

    def remove_widget(self, *args, **kwargs):
        if self.container:
            return self.container.remove_widget(*args, **kwargs)
        return super(DropDown, self).remove_widget(*args, **kwargs)

    def clear_widgets(self, *args, **kwargs):
        if self.container:
            return self.container.clear_widgets(*args, **kwargs)
        return super(DropDown, self).clear_widgets(*args, **kwargs)

    def on_motion(self, etype, me):
        super().on_motion(etype, me)
        return True

    def on_touch_down(self, touch):
        self._touch_started_inside = self.collide_point(*touch.pos)
        if not self.auto_dismiss or self._touch_started_inside:
            super(DropDown, self).on_touch_down(touch)
        return True

    def on_touch_move(self, touch):
        if not self.auto_dismiss or self._touch_started_inside:
            super(DropDown, self).on_touch_move(touch)
        return True

    def on_touch_up(self, touch):
        if self.auto_dismiss and self._touch_started_inside is False:
            self.dismiss()
        else:
            super(DropDown, self).on_touch_up(touch)
        self._touch_started_inside = None
        return True

    def _reposition(self, *largs):
        win = self._win
        if not win:
            return
        widget = self.attach_to
        if not widget or not widget.get_parent_window():
            return
        wx, wy = widget.to_window(*widget.pos)
        wright, wtop = widget.to_window(widget.right, widget.top)
        if self.auto_width:
            self.width = wright - wx
        x = wx
        if x + self.width > win.width:
            x = win.width - self.width
        if x < 0:
            x = 0
        self.x = x
        if self.max_height is not None:
            height = min(self.max_height, self.container.minimum_height)
        else:
            height = self.container.minimum_height
        h_bottom = wy - height
        h_top = win.height - (wtop + height)
        if h_bottom > 0:
            self.top = wy
            self.height = height
        elif h_top > 0:
            self.y = wtop
            self.height = height
        elif h_top < h_bottom:
            self.top = self.height = wy
        else:
            self.y = wtop
            self.height = win.height - wtop
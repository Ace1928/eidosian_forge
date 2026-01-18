from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
class TouchRippleButtonBehavior(TouchRippleBehavior):
    """
    This `mixin <https://en.wikipedia.org/wiki/Mixin>`_ class provides
    a similar behavior to :class:`~kivy.uix.behaviors.button.ButtonBehavior`
    but provides touch ripple animation instead of button pressed/released as
    visual effect.

    :Events:
        `on_press`
            Fired when the button is pressed.
        `on_release`
            Fired when the button is released (i.e. the touch/click that
            pressed the button goes away).
    """
    last_touch = ObjectProperty(None)
    'Contains the last relevant touch received by the Button. This can\n    be used in `on_press` or `on_release` in order to know which touch\n    dispatched the event.\n\n    :attr:`last_touch` is a :class:`~kivy.properties.ObjectProperty` and\n    defaults to `None`.\n    '
    always_release = BooleanProperty(False)
    'This determines whether or not the widget fires an `on_release` event if\n    the touch_up is outside the widget.\n\n    :attr:`always_release` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to `False`.\n    '

    def __init__(self, **kwargs):
        self.register_event_type('on_press')
        self.register_event_type('on_release')
        super(TouchRippleButtonBehavior, self).__init__(**kwargs)

    def on_touch_down(self, touch):
        if super(TouchRippleButtonBehavior, self).on_touch_down(touch):
            return True
        if touch.is_mouse_scrolling:
            return False
        if not self.collide_point(touch.x, touch.y):
            return False
        if self in touch.ud:
            return False
        touch.grab(self)
        touch.ud[self] = True
        self.last_touch = touch
        self.ripple_show(touch)
        self.dispatch('on_press')
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            return True
        if super(TouchRippleButtonBehavior, self).on_touch_move(touch):
            return True
        return self in touch.ud

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super(TouchRippleButtonBehavior, self).on_touch_up(touch)
        assert self in touch.ud
        touch.ungrab(self)
        self.last_touch = touch
        if self.disabled:
            return
        self.ripple_fade()
        if not self.always_release and (not self.collide_point(*touch.pos)):
            return

        def defer_release(dt):
            self.dispatch('on_release')
        Clock.schedule_once(defer_release, self.ripple_duration_out)
        return True

    def on_disabled(self, instance, value):
        if value:
            self.ripple_fade()
        return super(TouchRippleButtonBehavior, self).on_disabled(instance, value)

    def on_press(self):
        pass

    def on_release(self):
        pass
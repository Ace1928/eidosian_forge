import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
class PushButton(WidgetBase):
    """Instance of a push button.

    Triggers the event 'on_press' when it is clicked by the mouse.
    Triggers the event 'on_release' when the mouse is released.
    """

    def __init__(self, x, y, pressed, depressed, hover=None, batch=None, group=None):
        """Create a push button.

        :Parameters:
            `x` : int
                X coordinate of the push button.
            `y` : int
                Y coordinate of the push button.
            `pressed` : `~pyglet.image.AbstractImage`
                Image to display when the button is pressed.
            `depresseed` : `~pyglet.image.AbstractImage`
                Image to display when the button isn't pressed.
            `hover` : `~pyglet.image.AbstractImage`
                Image to display when the button is being hovered over.
            `batch` : `~pyglet.graphics.Batch`
                Optional batch to add the push button to.
            `group` : `~pyglet.graphics.Group`
                Optional parent group of the push button.
        """
        super().__init__(x, y, depressed.width, depressed.height)
        self._pressed_img = pressed
        self._depressed_img = depressed
        self._hover_img = hover or depressed
        self._batch = batch or pyglet.graphics.Batch()
        self._user_group = group
        bg_group = Group(order=0, parent=group)
        self._sprite = pyglet.sprite.Sprite(self._depressed_img, x, y, batch=batch, group=bg_group)
        self._pressed = False

    def _update_position(self):
        self._sprite.position = (self._x, self._y, 0)

    @property
    def value(self):
        return self._pressed

    @value.setter
    def value(self, value):
        assert type(value) is bool, "This Widget's value must be True or False."
        self._pressed = value
        self._sprite.image = self._pressed_img if self._pressed else self._depressed_img

    def update_groups(self, order):
        self._sprite.group = Group(order=order + 1, parent=self._user_group)

    def on_mouse_press(self, x, y, buttons, modifiers):
        if not self.enabled or not self._check_hit(x, y):
            return
        self._sprite.image = self._pressed_img
        self._pressed = True
        self.dispatch_event('on_press')

    def on_mouse_release(self, x, y, buttons, modifiers):
        if not self.enabled or not self._pressed:
            return
        self._sprite.image = self._hover_img if self._check_hit(x, y) else self._depressed_img
        self._pressed = False
        self.dispatch_event('on_release')

    def on_mouse_motion(self, x, y, dx, dy):
        if not self.enabled or self._pressed:
            return
        self._sprite.image = self._hover_img if self._check_hit(x, y) else self._depressed_img

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self.enabled or self._pressed:
            return
        self._sprite.image = self._hover_img if self._check_hit(x, y) else self._depressed_img

    def on_press(self):
        """Event: Dispatched when the button is clicked."""

    def on_release(self):
        """Event: Dispatched when the button is released."""
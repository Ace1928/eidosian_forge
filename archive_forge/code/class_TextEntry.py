import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
class TextEntry(WidgetBase):
    """Instance of a text entry widget. Allows the user to enter and submit text.
    
    Triggers the event 'on_commit', when the user hits the Enter or Return key.
    The current text string is passed along with the event.
    """

    def __init__(self, text, x, y, width, color=(255, 255, 255, 255), text_color=(0, 0, 0, 255), caret_color=(0, 0, 0, 255), batch=None, group=None):
        """Create a text entry widget.

        :Parameters:
            `text` : str
                Initial text to display.
            `x` : int
                X coordinate of the text entry widget.
            `y` : int
                Y coordinate of the text entry widget.
            `width` : int
                The width of the text entry widget.
            `color` : (int, int, int, int)
                The color of the outline box in RGBA format.
            `text_color` : (int, int, int, int)
                The color of the text in RGBA format.
            `caret_color` : (int, int, int, int)
                The color of the caret when it is visible in RGBA or RGB
                format.
            `batch` : `~pyglet.graphics.Batch`
                Optional batch to add the text entry widget to.
            `group` : `~pyglet.graphics.Group`
                Optional parent group of text entry widget.
        """
        self._doc = pyglet.text.document.UnformattedDocument(text)
        self._doc.set_style(0, len(self._doc.text), dict(color=text_color))
        font = self._doc.get_font()
        height = font.ascent - font.descent
        self._user_group = group
        bg_group = Group(order=0, parent=group)
        fg_group = Group(order=1, parent=group)
        self._pad = p = 2
        self._outline = pyglet.shapes.Rectangle(x - p, y - p, width + p + p, height + p + p, color[:3], batch, bg_group)
        self._outline.opacity = color[3]
        self._layout = IncrementalTextLayout(self._doc, width, height, multiline=False, batch=batch, group=fg_group)
        self._layout.x = x
        self._layout.y = y
        self._caret = Caret(self._layout, color=caret_color)
        self._caret.visible = False
        self._focus = False
        super().__init__(x, y, width, height)

    def _update_position(self):
        self._layout.position = (self._x, self._y, 0)
        self._outline.position = (self._x - self._pad, self._y - self._pad)

    @property
    def value(self):
        return self._doc.text

    @value.setter
    def value(self, value):
        assert type(value) is str, "This Widget's value must be a string."
        self._doc.text = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._layout.width = value
        self._outline.width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._layout.height = value
        self._outline.height = value

    @property
    def focus(self) -> bool:
        return self._focus

    @focus.setter
    def focus(self, value: bool) -> None:
        self._set_focus(value)

    def _check_hit(self, x, y):
        return self._x < x < self._x + self._width and self._y < y < self._y + self._height

    def _set_focus(self, value):
        self._focus = value
        self._caret.visible = value
        self._caret.layout = self._layout

    def update_groups(self, order):
        self._outline.group = Group(order=order + 1, parent=self._user_group)
        self._layout.group = Group(order=order + 2, parent=self._user_group)

    def on_mouse_motion(self, x, y, dx, dy):
        if not self.enabled:
            return

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self.enabled:
            return
        if self._focus:
            self._caret.on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_press(self, x, y, buttons, modifiers):
        if not self.enabled:
            return
        if self._check_hit(x, y):
            self._set_focus(True)
            self._caret.on_mouse_press(x, y, buttons, modifiers)
        else:
            self._set_focus(False)

    def on_text(self, text):
        if not self.enabled:
            return
        if self._focus:
            if text in ('\r', '\n'):
                self.dispatch_event('on_commit', self._layout.document.text)
                self._set_focus(False)
                return
            self._caret.on_text(text)

    def on_text_motion(self, motion):
        if not self.enabled:
            return
        if self._focus:
            self._caret.on_text_motion(motion)

    def on_text_motion_select(self, motion):
        if not self.enabled:
            return
        if self._focus:
            self._caret.on_text_motion_select(motion)

    def on_commit(self, text: str):
        """Event: dispatches the current text when commited via Enter/Return key."""
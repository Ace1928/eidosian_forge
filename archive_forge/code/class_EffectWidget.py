from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class EffectWidget(RelativeLayout):
    """
    Widget with the ability to apply a series of graphical effects to
    its children. See the module documentation for more information on
    setting effects and creating your own.
    """
    background_color = ListProperty((0, 0, 0, 0))
    'This defines the background color to be used for the fbo in the\n    EffectWidget.\n\n    :attr:`background_color` is a :class:`ListProperty` defaults to\n    (0, 0, 0, 0)\n    '
    texture = ObjectProperty(None)
    'The output texture of the final :class:`~kivy.graphics.Fbo` after\n    all effects have been applied.\n\n    texture is an :class:`~kivy.properties.ObjectProperty` and defaults\n    to None.\n    '
    effects = ListProperty([])
    'List of all the effects to be applied. These should all be\n    instances or subclasses of :class:`EffectBase`.\n\n    effects is a :class:`ListProperty` and defaults to [].\n    '
    fbo_list = ListProperty([])
    '(internal) List of all the fbos that are being used to apply\n    the effects.\n\n    fbo_list is a :class:`ListProperty` and defaults to [].\n    '
    _bound_effects = ListProperty([])
    '(internal) List of effect classes that have been given an fbo to\n    manage. This is necessary so that the fbo can be removed if the\n    effect is no longer in use.\n\n    _bound_effects is a :class:`ListProperty` and defaults to [].\n    '

    def __init__(self, **kwargs):
        EventLoop.ensure_window()
        self.canvas = RenderContext(use_parent_projection=True, use_parent_modelview=True)
        with self.canvas:
            self.fbo = Fbo(size=self.size)
        with self.fbo.before:
            PushMatrix()
        with self.fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            self._background_color = Color(*self.background_color)
            self.fbo_rectangle = Rectangle(size=self.size)
        with self.fbo.after:
            PopMatrix()
        super(EffectWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self._update_glsl, 0)
        fbind = self.fbind
        fbo_setup = self.refresh_fbo_setup
        fbind('size', fbo_setup)
        fbind('effects', fbo_setup)
        fbind('background_color', self._refresh_background_color)
        self.refresh_fbo_setup()
        self._refresh_background_color()

    def _refresh_background_color(self, *args):
        self._background_color.rgba = self.background_color

    def _update_glsl(self, *largs):
        """(internal) Passes new time and resolution uniform
        variables to the shader.
        """
        time = Clock.get_boottime()
        resolution = [float(size) for size in self.size]
        self.canvas['time'] = time
        self.canvas['resolution'] = resolution
        for fbo in self.fbo_list:
            fbo['time'] = time
            fbo['resolution'] = resolution

    def refresh_fbo_setup(self, *args):
        """(internal) Creates and assigns one :class:`~kivy.graphics.Fbo`
        per effect, and makes sure all sizes etc. are correct and
        consistent.
        """
        while len(self.fbo_list) < len(self.effects):
            with self.canvas:
                new_fbo = EffectFbo(size=self.size)
            with new_fbo:
                ClearColor(0, 0, 0, 0)
                ClearBuffers()
                Color(1, 1, 1, 1)
                new_fbo.texture_rectangle = Rectangle(size=self.size)
                new_fbo.texture_rectangle.size = self.size
            self.fbo_list.append(new_fbo)
        while len(self.fbo_list) > len(self.effects):
            old_fbo = self.fbo_list.pop()
            self.canvas.remove(old_fbo)
        for effect in self._bound_effects:
            if effect not in self.effects:
                effect.fbo = None
        self._bound_effects = self.effects
        self.fbo.size = self.size
        self.fbo_rectangle.size = self.size
        for i in range(len(self.fbo_list)):
            self.fbo_list[i].size = self.size
            self.fbo_list[i].texture_rectangle.size = self.size
        if len(self.fbo_list) == 0:
            self.texture = self.fbo.texture
            return
        for i in range(1, len(self.fbo_list)):
            fbo = self.fbo_list[i]
            fbo.texture_rectangle.texture = self.fbo_list[i - 1].texture
        for effect, fbo in zip(self.effects, self.fbo_list):
            effect.fbo = fbo
        self.fbo_list[0].texture_rectangle.texture = self.fbo.texture
        self.texture = self.fbo_list[-1].texture
        for fbo in self.fbo_list:
            fbo.draw()
        self.fbo.draw()

    def add_widget(self, *args, **kwargs):
        c = self.canvas
        self.canvas = self.fbo
        super(EffectWidget, self).add_widget(*args, **kwargs)
        self.canvas = c

    def remove_widget(self, *args, **kwargs):
        c = self.canvas
        self.canvas = self.fbo
        super(EffectWidget, self).remove_widget(*args, **kwargs)
        self.canvas = c

    def clear_widgets(self, *args, **kwargs):
        c = self.canvas
        self.canvas = self.fbo
        super(EffectWidget, self).clear_widgets(*args, **kwargs)
        self.canvas = c
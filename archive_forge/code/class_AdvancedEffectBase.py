from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class AdvancedEffectBase(EffectBase):
    """An :class:`EffectBase` with additional behavior to easily
    set and update uniform variables in your shader.

    This class is provided for convenience when implementing your own
    effects: it is not used by any of those provided with Kivy.

    In addition to your base glsl string that must be provided as
    normal, the :class:`AdvancedEffectBase` has an extra property
    :attr:`uniforms`, a dictionary of name-value pairs. Whenever
    a value is changed, the new value for the uniform variable is
    uploaded to the shader.

    You must still manually declare your uniform variables at the top
    of your glsl string.
    """
    uniforms = DictProperty({})
    'A dictionary of uniform variable names and their values. These\n    are automatically uploaded to the :attr:`fbo` shader if appropriate.\n\n    uniforms is a :class:`~kivy.properties.DictProperty` and\n    defaults to {}.\n    '

    def __init__(self, *args, **kwargs):
        super(AdvancedEffectBase, self).__init__(*args, **kwargs)
        self.fbind('uniforms', self._update_uniforms)

    def _update_uniforms(self, *args):
        if self.fbo is None:
            return
        for key, value in self.uniforms.items():
            self.fbo[key] = value

    def set_fbo_shader(self, *args):
        super(AdvancedEffectBase, self).set_fbo_shader(*args)
        self._update_uniforms()
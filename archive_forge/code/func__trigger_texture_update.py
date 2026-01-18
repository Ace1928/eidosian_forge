from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.core.text import Label as CoreLabel, DEFAULT_FONT
from kivy.core.text.markup import MarkupLabel as CoreMarkupLabel
from kivy.properties import StringProperty, OptionProperty, \
from kivy.utils import get_hex_from_color
def _trigger_texture_update(self, name=None, source=None, value=None):
    if name == 'markup':
        self._create_label()
    if source:
        if name == 'text':
            self._label.text = value
        elif name == 'text_size':
            self._label.usersize = value
        elif name == 'font_size':
            self._label.options[name] = value
        elif name == 'disabled_color' and self.disabled:
            self._label.options['color'] = value
        elif name == 'disabled_outline_color' and self.disabled:
            self._label.options['outline_color'] = value
        elif name == 'disabled':
            self._label.options['color'] = self.disabled_color if value else self.color
            self._label.options['outline_color'] = self.disabled_outline_color if value else self.outline_color
        elif name == 'padding_x':
            self._label.options['padding'][::2] = [value] * 2
        elif name == 'padding_y':
            self._label.options['padding'][1::2] = [value] * 2
        else:
            self._label.options[name] = value
    self._trigger_texture()
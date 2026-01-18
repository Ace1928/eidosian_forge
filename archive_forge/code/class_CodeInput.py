from pygments import highlight
from pygments import lexers
from pygments import styles
from pygments.formatters import BBCodeFormatter
from kivy.uix.textinput import TextInput
from kivy.core.text.markup import MarkupLabel as Label
from kivy.cache import Cache
from kivy.properties import ObjectProperty, OptionProperty
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.behaviors import CodeNavigationBehavior
class CodeInput(CodeNavigationBehavior, TextInput):
    """CodeInput class, used for displaying highlighted code.
    """
    lexer = ObjectProperty(None)
    'This holds the selected Lexer used by pygments to highlight the code.\n\n\n    :attr:`lexer` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to `PythonLexer`.\n    '
    style_name = OptionProperty('default', options=list(styles.get_all_styles()))
    "Name of the pygments style to use for formatting.\n\n    :attr:`style_name` is an :class:`~kivy.properties.OptionProperty`\n    and defaults to ``'default'``.\n\n    "
    style = ObjectProperty(None)
    'The pygments style object to use for formatting.\n\n    When ``style_name`` is set, this will be changed to the\n    corresponding style object.\n\n    :attr:`style` is a :class:`~kivy.properties.ObjectProperty` and\n    defaults to ``None``\n\n    '

    def __init__(self, **kwargs):
        stylename = kwargs.get('style_name', 'default')
        style = kwargs['style'] if 'style' in kwargs else styles.get_style_by_name(stylename)
        self.formatter = BBCodeFormatter(style=style)
        self.lexer = lexers.PythonLexer()
        self.text_color = '#000000'
        self._label_cached = Label()
        self.use_text_color = True
        super(CodeInput, self).__init__(**kwargs)
        self._line_options = kw = self._get_line_options()
        self._label_cached = Label(**kw)
        text_color = kwargs.get('foreground_color')
        if text_color:
            self.text_color = get_hex_from_color(text_color)
        self.use_text_color = False
        self.foreground_color = [1, 1, 1, 0.999]
        if not kwargs.get('background_color'):
            self.background_color = [0.9, 0.92, 0.92, 1]

    def on_style_name(self, *args):
        self.style = styles.get_style_by_name(self.style_name)
        self.background_color = get_color_from_hex(self.style.background_color)
        self._trigger_refresh_text()

    def on_style(self, *args):
        self.formatter = BBCodeFormatter(style=self.style)
        self._trigger_update_graphics()

    def _create_line_label(self, text, hint=False):
        ntext = text.replace(u'\n', u'').replace(u'\t', u' ' * self.tab_width)
        if self.password and (not hint):
            ntext = u'*' * len(ntext)
        ntext = self._get_bbcode(ntext)
        kw = self._get_line_options()
        cid = u'{}\x00{}\x00{}'.format(text, self.password, kw)
        texture = Cache_get('textinput.label', cid)
        if texture is None:
            label = Label(text=ntext, **kw)
            if text.find(u'\n') > 0:
                label.text = u''
            else:
                label.text = ntext
            label.refresh()
            texture = label.texture
            Cache_append('textinput.label', cid, texture)
            label.text = ''
        return texture

    def _get_line_options(self):
        kw = super(CodeInput, self)._get_line_options()
        kw['markup'] = True
        kw['valign'] = 'top'
        kw['codeinput'] = repr(self.lexer)
        return kw

    def _get_text_width(self, text, tab_width, _label_cached):
        cid = u'{}\x00{}\x00{}'.format(text, self.password, self._get_line_options())
        width = Cache_get('textinput.width', cid)
        if width is not None:
            return width
        lbl = self._create_line_label(text)
        width = lbl.width
        Cache_append('textinput.width', cid, width)
        return width

    def _get_bbcode(self, ntext):
        try:
            ntext[0]
            ntext = ntext.replace(u'[', u'\x01').replace(u']', u'\x02')
            ntext = highlight(ntext, self.lexer, self.formatter)
            ntext = ntext.replace(u'\x01', u'&bl;').replace(u'\x02', u'&br;')
            ntext = ''.join((u'[color=', str(self.text_color), u']', ntext, u'[/color]'))
            ntext = ntext.replace(u'\n', u'')
            ntext = ntext.replace(u'[u]', '').replace(u'[/u]', '')
            return ntext
        except IndexError:
            return ''

    def _cursor_offset(self):
        """Get the cursor x offset on the current line
        """
        offset = 0
        try:
            if self.cursor_col:
                offset = self._get_text_width(self._lines[self.cursor_row][:self.cursor_col])
                return offset
        except:
            pass
        finally:
            return offset

    def on_lexer(self, instance, value):
        self._trigger_refresh_text()

    def on_foreground_color(self, instance, text_color):
        if not self.use_text_color:
            self.use_text_color = True
            return
        self.text_color = get_hex_from_color(text_color)
        self.use_text_color = False
        self.foreground_color = (1, 1, 1, 0.999)
        self._trigger_refresh_text()
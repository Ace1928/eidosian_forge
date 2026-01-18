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
import os
from os.path import dirname, join, exists, abspath
from kivy.clock import Clock
from kivy.compat import PY2
from kivy.properties import ObjectProperty, NumericProperty, \
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage, Image
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.anchorlayout import AnchorLayout
from kivy.animation import Animation
from kivy.logger import Logger
from docutils.parsers import rst
from docutils.parsers.rst import roles
from docutils import nodes, frontend, utils
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.roles import set_classes
class RstDocument(ScrollView):
    """Base widget used to store an Rst document. See module documentation for
    more information.
    """
    source = StringProperty(None)
    'Filename of the RST document.\n\n    :attr:`source` is a :class:`~kivy.properties.StringProperty` and\n    defaults to None.\n    '
    source_encoding = StringProperty('utf-8')
    'Encoding to be used for the :attr:`source` file.\n\n    :attr:`source_encoding` is a :class:`~kivy.properties.StringProperty` and\n    defaults to `utf-8`.\n\n    .. Note::\n        It is your responsibility to ensure that the value provided is a\n        valid codec supported by python.\n    '
    source_error = OptionProperty('strict', options=('strict', 'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplac'))
    "Error handling to be used while encoding the :attr:`source` file.\n\n    :attr:`source_error` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to `strict`. Can be one of 'strict', 'ignore', 'replace',\n    'xmlcharrefreplace' or 'backslashreplac'.\n    "
    text = StringProperty(None)
    'RST markup text of the document.\n\n    :attr:`text` is a :class:`~kivy.properties.StringProperty` and defaults to\n    None.\n    '
    document_root = StringProperty(None)
    'Root path where :doc: will search for rst documents. If no path is\n    given, it will use the directory of the first loaded source file.\n\n    :attr:`document_root` is a :class:`~kivy.properties.StringProperty` and\n    defaults to None.\n    '
    base_font_size = NumericProperty(31)
    'Font size for the biggest title, 31 by default. All other font sizes are\n    derived from this.\n\n    .. versionadded:: 1.8.0\n    '
    show_errors = BooleanProperty(False)
    'Indicate whether RST parsers errors should be shown on the screen\n    or not.\n\n    :attr:`show_errors` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '

    def _get_bgc(self):
        return get_color_from_hex(self.colors.background)

    def _set_bgc(self, value):
        self.colors.background = get_hex_from_color(value)[1:]
    background_color = AliasProperty(_get_bgc, _set_bgc, bind=('colors',), cache=True)
    "Specifies the background_color to be used for the RstDocument.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`background_color` is an :class:`~kivy.properties.AliasProperty`\n    for colors['background'].\n    "
    colors = DictProperty({'background': 'e5e6e9ff', 'link': 'ce5c00ff', 'paragraph': '202020ff', 'title': '204a87ff', 'bullet': '000000ff'})
    'Dictionary of all the colors used in the RST rendering.\n\n    .. warning::\n\n        This dictionary is needs special handling. You also need to call\n        :meth:`RstDocument.render` if you change them after loading.\n\n    :attr:`colors` is a :class:`~kivy.properties.DictProperty`.\n    '
    title = StringProperty('')
    "Title of the current document.\n\n    :attr:`title` is a :class:`~kivy.properties.StringProperty` and defaults to\n    ''. It is read-only.\n    "
    toctrees = DictProperty({})
    "Toctree of all loaded or preloaded documents. This dictionary is filled\n    when a rst document is explicitly loaded or where :meth:`preload` has been\n    called.\n\n    If the document has no filename, e.g. when the document is loaded from a\n    text file, the key will be ''.\n\n    :attr:`toctrees` is a :class:`~kivy.properties.DictProperty` and defaults\n    to {}.\n    "
    underline_color = StringProperty('204a9699')
    "underline color of the titles, expressed in html color notation\n\n    :attr:`underline_color` is a\n    :class:`~kivy.properties.StringProperty` and defaults to '204a9699'.\n\n    .. versionadded: 1.9.0\n    "
    content = ObjectProperty(None)
    scatter = ObjectProperty(None)
    anchors_widgets = ListProperty([])
    refs_assoc = DictProperty({})

    def __init__(self, **kwargs):
        self._trigger_load = Clock.create_trigger(self._load_from_text, -1)
        self._parser = rst.Parser()
        self._settings = frontend.OptionParser(components=(rst.Parser,)).get_default_values()
        super(RstDocument, self).__init__(**kwargs)

    def on_source(self, instance, value):
        if not value:
            return
        if self.document_root is None:
            self.document_root = abspath(dirname(value))
        self._load_from_source()

    def on_text(self, instance, value):
        self._trigger_load()

    def render(self):
        """Force document rendering.
        """
        self._load_from_text()

    def resolve_path(self, filename):
        """Get the path for this filename. If the filename doesn't exist,
        it returns the document_root + filename.
        """
        if exists(filename):
            return filename
        return join(self.document_root, filename)

    def preload(self, filename, encoding='utf-8', errors='strict'):
        """Preload a rst file to get its toctree and its title.

        The result will be stored in :attr:`toctrees` with the ``filename`` as
        key.
        """
        with open(filename, 'rb') as fd:
            text = fd.read().decode(encoding, errors)
        document = utils.new_document('Document', self._settings)
        self._parser.parse(text, document)
        visitor = _ToctreeVisitor(document)
        document.walkabout(visitor)
        self.toctrees[filename] = visitor.toctree
        return text

    def _load_from_source(self):
        filename = self.resolve_path(self.source)
        self.text = self.preload(filename, self.source_encoding, self.source_error)

    def _load_from_text(self, *largs):
        try:
            self.content.clear_widgets()
            self.anchors_widgets = []
            self.refs_assoc = {}
            document = utils.new_document('Document', self._settings)
            text = self.text
            if PY2 and type(text) is str:
                text = text.decode('utf-8')
            self._parser.parse(text, document)
            visitor = _Visitor(self, document)
            document.walkabout(visitor)
            self.title = visitor.title or 'No title'
        except:
            Logger.exception('Rst: error while loading text')

    def on_ref_press(self, node, ref):
        self.goto(ref)

    def goto(self, ref, *largs):
        """Scroll to the reference. If it's not found, nothing will be done.

        For this text::

            .. _myref:

            This is something I always wanted.

        You can do::

            from kivy.clock import Clock
            from functools import partial

            doc = RstDocument(...)
            Clock.schedule_once(partial(doc.goto, 'myref'), 0.1)

        .. note::

            It is preferable to delay the call of the goto if you just loaded
            the document because the layout might not be finished or the
            size of the RstDocument has not yet been determined. In
            either case, the calculation of the scrolling would be
            wrong.

            You can, however, do a direct call if the document is already
            loaded.

        .. versionadded:: 1.3.0
        """
        if ref.endswith('.rst'):
            self.source = ref
            return
        ref = self.refs_assoc.get(ref, ref)
        ax = ay = None
        for node in self.anchors_widgets:
            if ref in node.anchors:
                ax, ay = node.anchors[ref]
                break
        if ax is None:
            return
        ax += node.x
        ay = node.top - ay
        sx, sy = (self.scatter.x, self.scatter.top)
        ay -= self.height
        dx, dy = self.convert_distance_to_scroll(0, ay)
        dy = max(0, min(1, dy))
        Animation(scroll_y=dy, d=0.25, t='in_out_expo').start(self)

    def add_anchors(self, node):
        self.anchors_widgets.append(node)
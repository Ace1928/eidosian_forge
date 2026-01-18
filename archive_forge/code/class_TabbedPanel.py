from functools import partial
from kivy.clock import Clock
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, OptionProperty, \
class TabbedPanel(GridLayout):
    """The TabbedPanel class. See module documentation for more information.
    """
    background_color = ColorProperty([1, 1, 1, 1])
    'Background color, in the format (r, g, b, a).\n\n    :attr:`background_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    border = ListProperty([16, 16, 16, 16])
    'Border used for :class:`~kivy.graphics.vertex_instructions.BorderImage`\n    graphics instruction, used itself for :attr:`background_image`.\n    Can be changed for a custom background.\n\n    It must be a list of four values: (bottom, right, top, left). Read the\n    BorderImage instructions for more information.\n\n    :attr:`border` is a :class:`~kivy.properties.ListProperty` and\n    defaults to (16, 16, 16, 16)\n    '
    background_image = StringProperty('atlas://data/images/defaulttheme/tab')
    "Background image of the main shared content object.\n\n    :attr:`background_image` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/tab'.\n    "
    background_disabled_image = StringProperty('atlas://data/images/defaulttheme/tab_disabled')
    "Background image of the main shared content object when disabled.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`background_disabled_image` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/tab'.\n    "
    strip_image = StringProperty('atlas://data/images/defaulttheme/action_view')
    'Background image of the tabbed strip.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`strip_image` is a :class:`~kivy.properties.StringProperty`\n    and defaults to a empty image.\n    '
    strip_border = ListProperty([4, 4, 4, 4])
    'Border to be used on :attr:`strip_image`.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`strip_border` is a :class:`~kivy.properties.ListProperty` and\n    defaults to [4, 4, 4, 4].\n    '
    _current_tab = ObjectProperty(None)

    def get_current_tab(self):
        return self._current_tab
    current_tab = AliasProperty(get_current_tab, None, bind=('_current_tab',))
    'Links to the currently selected or active tab.\n\n    .. versionadded:: 1.4.0\n\n    :attr:`current_tab` is an :class:`~kivy.AliasProperty`, read-only.\n    '
    tab_pos = OptionProperty('top_left', options=('left_top', 'left_mid', 'left_bottom', 'top_left', 'top_mid', 'top_right', 'right_top', 'right_mid', 'right_bottom', 'bottom_left', 'bottom_mid', 'bottom_right'))
    "Specifies the position of the tabs relative to the content.\n    Can be one of: `left_top`, `left_mid`, `left_bottom`, `top_left`,\n    `top_mid`, `top_right`, `right_top`, `right_mid`, `right_bottom`,\n    `bottom_left`, `bottom_mid`, `bottom_right`.\n\n    :attr:`tab_pos` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to 'top_left'.\n    "
    tab_height = NumericProperty('40dp')
    'Specifies the height of the tab header.\n\n    :attr:`tab_height` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 40.\n    '
    tab_width = NumericProperty('100dp', allownone=True)
    'Specifies the width of the tab header.\n\n    :attr:`tab_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 100.\n    '
    bar_width = NumericProperty('2dp')
    'Width of the horizontal scroll bar. The width is interpreted\n    as a height for the horizontal bar.\n\n    .. versionadded:: 2.2.0\n\n    :attr:`bar_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 2.\n    '
    scroll_type = OptionProperty(['content'], options=(['content'], ['bars'], ['bars', 'content'], ['content', 'bars']))
    "Sets the type of scrolling to use for the content of the scrollview.\n    Available options are: ['content'], ['bars'], ['bars', 'content'].\n\n    .. versionadded:: 2.2.0\n\n    :attr:`scroll_type` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to ['content'].\n    "
    do_default_tab = BooleanProperty(True)
    "Specifies whether a default_tab head is provided.\n\n    .. versionadded:: 1.5.0\n\n    :attr:`do_default_tab` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to 'True'.\n    "
    default_tab_text = StringProperty('Default tab')
    "Specifies the text displayed on the default tab header.\n\n    :attr:`default_tab_text` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'default tab'.\n    "
    default_tab_cls = ObjectProperty(TabbedPanelHeader)
    'Specifies the class to use for the styling of the default tab.\n\n    .. versionadded:: 1.4.0\n\n    .. warning::\n        `default_tab_cls` should be subclassed from `TabbedPanelHeader`\n\n    :attr:`default_tab_cls` is an :class:`~kivy.properties.ObjectProperty`\n    and defaults to `TabbedPanelHeader`. If you set a string, the\n    :class:`~kivy.factory.Factory` will be used to resolve the class.\n\n    .. versionchanged:: 1.8.0\n        The :class:`~kivy.factory.Factory` will resolve the class if a string\n        is set.\n    '

    def get_tab_list(self):
        if self._tab_strip:
            return self._tab_strip.children
        return 1.0
    tab_list = AliasProperty(get_tab_list, None)
    'List of all the tab headers.\n\n    :attr:`tab_list` is an :class:`~kivy.properties.AliasProperty` and is\n    read-only.\n    '
    content = ObjectProperty(None)
    "This is the object holding (current_tab's content is added to this)\n    the content of the current tab. To Listen to the changes in the content\n    of the current tab, you should bind to current_tabs `content` property.\n\n    :attr:`content` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to 'None'.\n    "
    _default_tab = ObjectProperty(None, allow_none=True)

    def get_def_tab(self):
        return self._default_tab

    def set_def_tab(self, new_tab):
        if not issubclass(new_tab.__class__, TabbedPanelHeader):
            raise TabbedPanelException('`default_tab_class` should be                subclassed from `TabbedPanelHeader`')
        if self._default_tab == new_tab:
            return
        oltab = self._default_tab
        self._default_tab = new_tab
        self.remove_widget(oltab)
        self._original_tab = None
        self.switch_to(new_tab)
        new_tab.state = 'down'
    default_tab = AliasProperty(get_def_tab, set_def_tab, bind=('_default_tab',))
    'Holds the default tab.\n\n    .. Note:: For convenience, the automatically provided default tab is\n              deleted when you change default_tab to something else.\n              As of 1.5.0, this behavior has been extended to every\n              `default_tab` for consistency and not just the automatically\n              provided one.\n\n    :attr:`default_tab` is an :class:`~kivy.properties.AliasProperty`.\n    '

    def get_def_tab_content(self):
        return self.default_tab.content

    def set_def_tab_content(self, *l):
        self.default_tab.content = l[0]
    default_tab_content = AliasProperty(get_def_tab_content, set_def_tab_content)
    'Holds the default tab content.\n\n    :attr:`default_tab_content` is an :class:`~kivy.properties.AliasProperty`.\n    '
    _update_top_ev = _update_tab_ev = _update_tabs_ev = None

    def __init__(self, **kwargs):
        self._childrens = []
        self._tab_layout = StripLayout(rows=1)
        self.rows = 1
        self._tab_strip = TabbedPanelStrip(tabbed_panel=self, rows=1, size_hint=(None, None), height=self.tab_height, width=self.tab_width)
        self._partial_update_scrollview = None
        self.content = TabbedPanelContent()
        self._current_tab = self._original_tab = self._default_tab = TabbedPanelHeader()
        super(TabbedPanel, self).__init__(**kwargs)
        self.fbind('size', self._reposition_tabs)
        if not self.do_default_tab:
            Clock.schedule_once(self._switch_to_first_tab)
            return
        self._setup_default_tab()
        self.switch_to(self.default_tab)

    def switch_to(self, header, do_scroll=False):
        """Switch to a specific panel header.

        .. versionchanged:: 1.10.0

        If used with `do_scroll=True`, it scrolls
        to the header's tab too.

        :meth:`switch_to` cannot be called from within the
        :class:`TabbedPanel` or its subclass' ``__init__`` method.
        If that is required, use the ``Clock`` to schedule it. See `discussion
        <https://github.com/kivy/kivy/issues/3493#issuecomment-121567969>`_
        for full example.
        """
        header_content = header.content
        self._current_tab.state = 'normal'
        header.state = 'down'
        self._current_tab = header
        self.clear_widgets()
        if header_content is None:
            return
        parent = header_content.parent
        if parent:
            parent.remove_widget(header_content)
        self.add_widget(header_content)
        if do_scroll:
            tabs = self._tab_strip
            tabs.parent.scroll_to(header)

    def clear_tabs(self, *l):
        self_tabs = self._tab_strip
        self_tabs.clear_widgets()
        if self.do_default_tab:
            self_default_tab = self._default_tab
            self_tabs.add_widget(self_default_tab)
            self_tabs.width = self_default_tab.width
        self._reposition_tabs()

    def add_widget(self, widget, *args, **kwargs):
        content = self.content
        if content is None:
            return
        parent = widget.parent
        if parent:
            parent.remove_widget(widget)
        if widget in (content, self._tab_layout):
            super(TabbedPanel, self).add_widget(widget, *args, **kwargs)
        elif isinstance(widget, TabbedPanelHeader):
            self_tabs = self._tab_strip
            self_tabs.add_widget(widget, *args, **kwargs)
            widget.group = '__tab%r__' % self_tabs.uid
            self.on_tab_width()
        else:
            widget.pos_hint = {'x': 0, 'top': 1}
            self._childrens.append(widget)
            content.disabled = self.current_tab.disabled
            content.add_widget(widget, *args, **kwargs)

    def remove_widget(self, widget, *args, **kwargs):
        content = self.content
        if content is None:
            return
        if widget in (content, self._tab_layout):
            super(TabbedPanel, self).remove_widget(widget, *args, **kwargs)
        elif isinstance(widget, TabbedPanelHeader):
            if not (self.do_default_tab and widget is self._default_tab):
                self_tabs = self._tab_strip
                self_tabs.width -= widget.width
                self_tabs.remove_widget(widget)
                if widget.state == 'down' and self.do_default_tab:
                    self._default_tab.on_release()
                self._reposition_tabs()
            else:
                Logger.info("TabbedPanel: default tab! can't be removed.\n" + 'Change `default_tab` to a different tab.')
        else:
            if widget in self._childrens:
                self._childrens.remove(widget)
            if widget in content.children:
                content.remove_widget(widget, *args, **kwargs)

    def clear_widgets(self, *args, **kwargs):
        if self.content:
            self.content.clear_widgets(*args, **kwargs)

    def on_strip_image(self, instance, value):
        if not self._tab_layout:
            return
        self._tab_layout.background_image = value

    def on_strip_border(self, instance, value):
        if not self._tab_layout:
            return
        self._tab_layout.border = value

    def on_do_default_tab(self, instance, value):
        if not value:
            dft = self.default_tab
            if dft in self.tab_list:
                self.remove_widget(dft)
                self._switch_to_first_tab()
                self._default_tab = self._current_tab
        else:
            self._current_tab.state = 'normal'
            self._setup_default_tab()

    def on_default_tab_text(self, *args):
        self._default_tab.text = self.default_tab_text

    def on_tab_width(self, *l):
        ev = self._update_tab_ev
        if ev is None:
            ev = self._update_tab_ev = Clock.create_trigger(self._update_tab_width, 0)
        ev()

    def on_tab_height(self, *l):
        self._tab_layout.height = self._tab_strip.height = self.tab_height
        self._reposition_tabs()

    def on_tab_pos(self, *l):
        self._reposition_tabs()

    def _setup_default_tab(self):
        if self._default_tab in self.tab_list:
            return
        content = self._default_tab.content
        _tabs = self._tab_strip
        cls = self.default_tab_cls
        if isinstance(cls, string_types):
            cls = Factory.get(cls)
        if not issubclass(cls, TabbedPanelHeader):
            raise TabbedPanelException('`default_tab_class` should be                subclassed from `TabbedPanelHeader`')
        if cls != TabbedPanelHeader:
            self._current_tab = self._original_tab = self._default_tab = cls()
        default_tab = self.default_tab
        if self._original_tab == self.default_tab:
            default_tab.text = self.default_tab_text
        default_tab.height = self.tab_height
        default_tab.group = '__tab%r__' % _tabs.uid
        default_tab.state = 'down'
        default_tab.width = self.tab_width if self.tab_width else 100
        default_tab.content = content
        tl = self.tab_list
        if default_tab not in tl:
            _tabs.add_widget(default_tab, len(tl))
        if default_tab.content:
            self.clear_widgets()
            self.add_widget(self.default_tab.content)
        else:
            Clock.schedule_once(self._load_default_tab_content)
        self._current_tab = default_tab

    def _switch_to_first_tab(self, *l):
        ltl = len(self.tab_list) - 1
        if ltl > -1:
            self._current_tab = dt = self._original_tab = self.tab_list[ltl]
            self.switch_to(dt)

    def _load_default_tab_content(self, dt):
        if self.default_tab:
            self.switch_to(self.default_tab)

    def _reposition_tabs(self, *l):
        ev = self._update_tabs_ev
        if ev is None:
            ev = self._update_tabs_ev = Clock.create_trigger(self._update_tabs, 0)
        ev()

    def _update_tabs(self, *l):
        self_content = self.content
        if not self_content:
            return
        tab_pos = self.tab_pos
        tab_layout = self._tab_layout
        tab_layout.clear_widgets()
        scrl_v = ScrollView(size_hint=(None, 1), always_overscroll=False, bar_width=self.bar_width, scroll_type=self.scroll_type)
        tabs = self._tab_strip
        parent = tabs.parent
        if parent:
            parent.remove_widget(tabs)
        scrl_v.add_widget(tabs)
        scrl_v.pos = (0, 0)
        self_update_scrollview = self._update_scrollview
        if self._partial_update_scrollview is not None:
            tabs.unbind(width=self._partial_update_scrollview)
        self._partial_update_scrollview = partial(self_update_scrollview, scrl_v)
        tabs.bind(width=self._partial_update_scrollview)
        super(TabbedPanel, self).clear_widgets()
        tab_height = self.tab_height
        widget_list = []
        tab_list = []
        pos_letter = tab_pos[0]
        if pos_letter == 'b' or pos_letter == 't':
            self.cols = 1
            self.rows = 2
            tab_layout.rows = 1
            tab_layout.cols = 3
            tab_layout.size_hint = (1, None)
            tab_layout.height = tab_height + tab_layout.padding[1] + tab_layout.padding[3] + dp(2)
            self_update_scrollview(scrl_v)
            if pos_letter == 'b':
                if tab_pos == 'bottom_mid':
                    tab_list = (Widget(), scrl_v, Widget())
                    widget_list = (self_content, tab_layout)
                else:
                    if tab_pos == 'bottom_left':
                        tab_list = (scrl_v, Widget(), Widget())
                    elif tab_pos == 'bottom_right':
                        tab_list = (Widget(), Widget(), scrl_v)
                    widget_list = (self_content, tab_layout)
            else:
                if tab_pos == 'top_mid':
                    tab_list = (Widget(), scrl_v, Widget())
                elif tab_pos == 'top_left':
                    tab_list = (scrl_v, Widget(), Widget())
                elif tab_pos == 'top_right':
                    tab_list = (Widget(), Widget(), scrl_v)
                widget_list = (tab_layout, self_content)
        elif pos_letter == 'l' or pos_letter == 'r':
            self.cols = 2
            self.rows = 1
            tab_layout.rows = 3
            tab_layout.cols = 1
            tab_layout.size_hint = (None, 1)
            tab_layout.width = tab_height
            scrl_v.height = tab_height
            self_update_scrollview(scrl_v)
            rotation = 90 if tab_pos[0] == 'l' else -90
            sctr = Scatter(do_translation=False, rotation=rotation, do_rotation=False, do_scale=False, size_hint=(None, None), auto_bring_to_front=False, size=scrl_v.size)
            sctr.add_widget(scrl_v)
            lentab_pos = len(tab_pos)
            if tab_pos[lentab_pos - 4:] == '_top':
                sctr.bind(pos=partial(self._update_top, sctr, 'top', None))
                tab_list = (sctr,)
            elif tab_pos[lentab_pos - 4:] == '_mid':
                sctr.bind(pos=partial(self._update_top, sctr, 'mid', scrl_v.width))
                tab_list = (Widget(), sctr, Widget())
            elif tab_pos[lentab_pos - 7:] == '_bottom':
                tab_list = (Widget(), Widget(), sctr)
            if pos_letter == 'l':
                widget_list = (tab_layout, self_content)
            else:
                widget_list = (self_content, tab_layout)
        add = tab_layout.add_widget
        for widg in tab_list:
            add(widg)
        add = self.add_widget
        for widg in widget_list:
            add(widg)

    def _update_tab_width(self, *l):
        if self.tab_width:
            for tab in self.tab_list:
                tab.size_hint_x = 1
            tsw = self.tab_width * len(self._tab_strip.children)
        else:
            tsw = 0
            for tab in self.tab_list:
                if tab.size_hint_x:
                    tab.size_hint_x = 1
                    tsw += 100
                else:
                    tsw += tab.width
        self._tab_strip.width = tsw
        self._reposition_tabs()

    def _update_top(self, *args):
        sctr, top, scrl_v_width, x, y = args
        ev = self._update_top_ev
        if ev is not None:
            ev.cancel()
        ev = self._update_top_ev = Clock.schedule_once(partial(self._updt_top, sctr, top, scrl_v_width), 0)

    def _updt_top(self, sctr, top, scrl_v_width, *args):
        if top[0] == 't':
            sctr.top = self.top
        else:
            sctr.top = self.top - (self.height - scrl_v_width) / 2

    def _update_scrollview(self, scrl_v, *l):
        self_tab_pos = self.tab_pos
        self_tabs = self._tab_strip
        if self_tab_pos[0] == 'b' or self_tab_pos[0] == 't':
            scrl_v.width = min(self.width, self_tabs.width)
            scrl_v.top += 1
            scrl_v.top -= 1
        else:
            scrl_v.width = min(self.height, self_tabs.width)
            self_tabs.pos = (0, 0)
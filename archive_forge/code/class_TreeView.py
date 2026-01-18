from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
class TreeView(Widget):
    """TreeView class. See module documentation for more information.

    :Events:
        `on_node_expand`: (node, )
            Fired when a node is being expanded
        `on_node_collapse`: (node, )
            Fired when a node is being collapsed
    """
    __events__ = ('on_node_expand', 'on_node_collapse')

    def __init__(self, **kwargs):
        self._trigger_layout = Clock.create_trigger(self._do_layout, -1)
        super(TreeView, self).__init__(**kwargs)
        tvlabel = TreeViewLabel(text='Root', is_open=True, level=0)
        for key, value in self.root_options.items():
            setattr(tvlabel, key, value)
        self._root = self.add_node(tvlabel, None)
        trigger = self._trigger_layout
        fbind = self.fbind
        fbind('pos', trigger)
        fbind('size', trigger)
        fbind('indent_level', trigger)
        fbind('indent_start', trigger)
        trigger()

    def add_node(self, node, parent=None):
        """Add a new node to the tree.

        :Parameters:
            `node`: instance of a :class:`TreeViewNode`
                Node to add into the tree
            `parent`: instance of a :class:`TreeViewNode`, defaults to None
                Parent node to attach the new node. If `None`, it is added to
                the :attr:`root` node.

        :returns:
            the node `node`.
        """
        if not isinstance(node, TreeViewNode):
            raise TreeViewException('The node must be a subclass of TreeViewNode')
        if parent is None and self._root:
            parent = self._root
        if parent:
            parent.is_leaf = False
            parent.nodes.append(node)
            node.parent_node = parent
            node.level = parent.level + 1
        node.fbind('size', self._trigger_layout)
        self._trigger_layout()
        return node

    def remove_node(self, node):
        """Removes a node from the tree.

        .. versionadded:: 1.0.7

        :Parameters:
            `node`: instance of a :class:`TreeViewNode`
                Node to remove from the tree. If `node` is :attr:`root`, it is
                not removed.
        """
        if not isinstance(node, TreeViewNode):
            raise TreeViewException('The node must be a subclass of TreeViewNode')
        parent = node.parent_node
        if parent is not None:
            if node == self._selected_node:
                node.is_selected = False
                self._selected_node = None
            nodes = parent.nodes
            if node in nodes:
                nodes.remove(node)
            parent.is_leaf = not bool(len(nodes))
            node.parent_node = None
            node.funbind('size', self._trigger_layout)
            self._trigger_layout()

    def on_node_expand(self, node):
        pass

    def on_node_collapse(self, node):
        pass

    def select_node(self, node):
        """Select a node in the tree.
        """
        if node.no_selection:
            return
        if self._selected_node:
            self._selected_node.is_selected = False
        node.is_selected = True
        self._selected_node = node

    def deselect_node(self, *args):
        """Deselect any selected node.

        .. versionadded:: 1.10.0
        """
        if self._selected_node:
            self._selected_node.is_selected = False
            self._selected_node = None

    def toggle_node(self, node):
        """Toggle the state of the node (open/collapsed).
        """
        node.is_open = not node.is_open
        if node.is_open:
            if self.load_func and (not node.is_loaded):
                self._do_node_load(node)
            self.dispatch('on_node_expand', node)
        else:
            self.dispatch('on_node_collapse', node)
        self._trigger_layout()

    def get_node_at_pos(self, pos):
        """Get the node at the position (x, y).
        """
        x, y = pos
        for node in self.iterate_open_nodes(self.root):
            if self.x <= x <= self.right and node.y <= y <= node.top:
                return node

    def iterate_open_nodes(self, node=None):
        """Generator to iterate over all the expended nodes starting from
        `node` and down. If `node` is `None`, the generator start with
        :attr:`root`.

        To get all the open nodes::

            treeview = TreeView()
            # ... add nodes ...
            for node in treeview.iterate_open_nodes():
                print(node)

        """
        if not node:
            node = self.root
        if self.hide_root and node is self.root:
            pass
        else:
            yield node
        if not node.is_open:
            return
        f = self.iterate_open_nodes
        for cnode in node.nodes:
            for ynode in f(cnode):
                yield ynode

    def iterate_all_nodes(self, node=None):
        """Generator to iterate over all nodes from `node` and down whether
        expanded or not. If `node` is `None`, the generator start with
        :attr:`root`.
        """
        if not node:
            node = self.root
        yield node
        f = self.iterate_all_nodes
        for cnode in node.nodes:
            for ynode in f(cnode):
                yield ynode

    def on_load_func(self, instance, value):
        if value:
            Clock.schedule_once(self._do_initial_load)

    def _do_initial_load(self, *largs):
        if not self.load_func:
            return
        self._do_node_load(None)

    def _do_node_load(self, node):
        gen = self.load_func(self, node)
        if node:
            node.is_loaded = True
        if not gen:
            return
        for cnode in gen:
            self.add_node(cnode, node)

    def on_root_options(self, instance, value):
        if not self.root:
            return
        for key, value in value.items():
            setattr(self.root, key, value)

    def _do_layout(self, *largs):
        self.clear_widgets()
        self._do_open_node(self.root)
        self._do_layout_node(self.root, 0, self.top)
        min_width = min_height = 0
        for count, node in enumerate(self.iterate_open_nodes(self.root)):
            node.odd = False if count % 2 else True
            min_width = max(min_width, node.right - self.x)
            min_height += node.height
        self.minimum_size = (min_width, min_height)

    def _do_open_node(self, node):
        if self.hide_root and node is self.root:
            height = 0
        else:
            self.add_widget(node)
            height = node.height
            if not node.is_open:
                return height
        for cnode in node.nodes:
            height += self._do_open_node(cnode)
        return height

    def _do_layout_node(self, node, level, y):
        if self.hide_root and node is self.root:
            level -= 1
        else:
            node.x = self.x + self.indent_start + level * self.indent_level
            node.top = y
            if node.size_hint_x:
                node.width = (self.width - (node.x - self.x)) * node.size_hint_x
            y -= node.height
            if not node.is_open:
                return y
        for cnode in node.nodes:
            y = self._do_layout_node(cnode, level + 1, y)
        return y

    def on_touch_down(self, touch):
        node = self.get_node_at_pos(touch.pos)
        if not node:
            return
        if node.disabled:
            return
        if node.x - self.indent_start <= touch.x < node.x:
            self.toggle_node(node)
        elif node.x <= touch.x:
            self.select_node(node)
            node.dispatch('on_touch_down', touch)
        return True
    _root = ObjectProperty(None)
    _selected_node = ObjectProperty(None, allownone=True)
    minimum_width = NumericProperty(0)
    'Minimum width needed to contain all children.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`minimum_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    minimum_height = NumericProperty(0)
    'Minimum height needed to contain all children.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`minimum_height` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    minimum_size = ReferenceListProperty(minimum_width, minimum_height)
    'Minimum size needed to contain all children.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`minimum_size` is a :class:`~kivy.properties.ReferenceListProperty`\n    of (:attr:`minimum_width`, :attr:`minimum_height`) properties.\n    '
    indent_level = NumericProperty('16dp')
    'Width used for the indentation of each level except the first level.\n\n    Computation of indent for each level of the tree is::\n\n        indent = indent_start + level * indent_level\n\n    :attr:`indent_level` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 16.\n    '
    indent_start = NumericProperty('24dp')
    'Indentation width of the level 0 / root node. This is mostly the initial\n    size to accommodate a tree icon (collapsed / expanded). See\n    :attr:`indent_level` for more information about the computation of level\n    indentation.\n\n    :attr:`indent_start` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 24.\n    '
    hide_root = BooleanProperty(False)
    'Use this property to show/hide the initial root node. If True, the root\n    node will be appear as a closed node.\n\n    :attr:`hide_root` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '

    def get_selected_node(self):
        return self._selected_node
    selected_node = AliasProperty(get_selected_node, None, bind=('_selected_node',))
    'Node selected by :meth:`TreeView.select_node` or by touch.\n\n    :attr:`selected_node` is a :class:`~kivy.properties.AliasProperty` and\n    defaults to None. It is read-only.\n    '

    def get_root(self):
        return self._root
    root = AliasProperty(get_root, None, bind=('_root',))
    "Root node.\n\n    By default, the root node widget is a :class:`TreeViewLabel` with text\n    'Root'. If you want to change the default options passed to the widget\n    creation, use the :attr:`root_options` property::\n\n        treeview = TreeView(root_options={\n            'text': 'Root directory',\n            'font_size': 15})\n\n    :attr:`root_options` will change the properties of the\n    :class:`TreeViewLabel` instance. However, you cannot change the class used\n    for root node yet.\n\n    :attr:`root` is an :class:`~kivy.properties.AliasProperty` and defaults to\n    None. It is read-only. However, the content of the widget can be changed.\n    "
    root_options = ObjectProperty({})
    'Default root options to pass for root widget. See :attr:`root` property\n    for more information about the usage of root_options.\n\n    :attr:`root_options` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to {}.\n    '
    load_func = ObjectProperty(None)
    "Callback to use for asynchronous loading. If set, asynchronous loading\n    will be automatically done. The callback must act as a Python generator\n    function, using yield to send data back to the treeview.\n\n    The callback should be in the format::\n\n        def callback(treeview, node):\n            for name in ('Item 1', 'Item 2'):\n                yield TreeViewLabel(text=name)\n\n    :attr:`load_func` is a :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    "
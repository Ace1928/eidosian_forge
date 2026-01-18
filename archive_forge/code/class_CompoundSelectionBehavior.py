from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
class CompoundSelectionBehavior(object):
    """The Selection behavior `mixin <https://en.wikipedia.org/wiki/Mixin>`_
    implements the logic behind keyboard and touch
    selection of selectable widgets managed by the derived widget. Please see
    the :mod:`compound selection behaviors module
    <kivy.uix.behaviors.compoundselection>` documentation
    for more information.

    .. versionadded:: 1.9.0
    """
    selected_nodes = ListProperty([])
    'The list of selected nodes.\n\n    .. note::\n\n        Multiple nodes can be selected right after one another e.g. using the\n        keyboard. When listening to :attr:`selected_nodes`, one should be\n        aware of this.\n\n    :attr:`selected_nodes` is a :class:`~kivy.properties.ListProperty` and\n    defaults to the empty list, []. It is read-only and should not be modified.\n    '
    touch_multiselect = BooleanProperty(False)
    'A special touch mode which determines whether touch events, as\n    processed by :meth:`select_with_touch`, will add the currently touched\n    node to the selection, or if it will clear the selection before adding the\n    node. This allows the selection of multiple nodes by simply touching them.\n\n    This is different from :attr:`multiselect` because when it is True,\n    simply touching an unselected node will select it, even if ctrl is not\n    pressed. If it is False, however, ctrl must be pressed in order to\n    add to the selection when :attr:`multiselect` is True.\n\n    .. note::\n\n        :attr:`multiselect`, when False, will disable\n        :attr:`touch_multiselect`.\n\n    :attr:`touch_multiselect` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to False.\n    '
    multiselect = BooleanProperty(False)
    'Determines whether multiple nodes can be selected. If enabled, keyboard\n    shift and ctrl selection, optionally combined with touch, for example, will\n    be able to select multiple widgets in the normally expected manner.\n    This dominates :attr:`touch_multiselect` when False.\n\n    :attr:`multiselect` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    touch_deselect_last = BooleanProperty(not _is_desktop)
    'Determines whether the last selected node can be deselected when\n    :attr:`multiselect` or :attr:`touch_multiselect` is False.\n\n    .. versionadded:: 1.10.0\n\n    :attr:`touch_deselect_last` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to True on mobile, False on desktop platforms.\n    '
    keyboard_select = BooleanProperty(True)
    'Determines whether the keyboard can be used for selection. If False,\n    keyboard inputs will be ignored.\n\n    :attr:`keyboard_select` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to True.\n    '
    page_count = NumericProperty(10)
    'Determines by how much the selected node is moved up or down, relative\n    to the position of the last selected node, when pageup (or pagedown) is\n    pressed.\n\n    :attr:`page_count` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 10.\n    '
    up_count = NumericProperty(1)
    'Determines by how much the selected node is moved up or down, relative\n    to the position of the last selected node, when the up (or down) arrow on\n    the keyboard is pressed.\n\n    :attr:`up_count` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 1.\n    '
    right_count = NumericProperty(1)
    'Determines by how much the selected node is moved up or down, relative\n    to the position of the last selected node, when the right (or left) arrow\n    on the keyboard is pressed.\n\n    :attr:`right_count` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 1.\n    '
    scroll_count = NumericProperty(0)
    'Determines by how much the selected node is moved up or down, relative\n    to the position of the last selected node, when the mouse scroll wheel is\n    scrolled.\n\n    :attr:`right_count` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    nodes_order_reversed = BooleanProperty(True)
    " (Internal) Indicates whether the order of the nodes as displayed top-\n    down is reversed compared to their order in :meth:`get_selectable_nodes`\n    (e.g. how the children property is reversed compared to how\n    it's displayed).\n    "
    text_entry_timeout = NumericProperty(1.0)
    'When typing characters in rapid succession (i.e. the time difference\n    since the last character is less than :attr:`text_entry_timeout`), the\n    keys get concatenated and the combined text is passed as the key argument\n    of :meth:`goto_node`.\n\n    .. versionadded:: 1.10.0\n    '
    _anchor = None
    _anchor_idx = 0
    _last_selected_node = None
    _last_node_idx = 0
    _ctrl_down = False
    _shift_down = False
    _word_filter = ''
    _last_key_time = 0
    _key_list = []
    _offset_counts = {}

    def __init__(self, **kwargs):
        super(CompoundSelectionBehavior, self).__init__(**kwargs)
        self._key_list = []

        def ensure_single_select(*l):
            if not self.multiselect and len(self.selected_nodes) > 1:
                self.clear_selection()
        update_counts = self._update_counts
        update_counts()
        fbind = self.fbind
        fbind('multiselect', ensure_single_select)
        fbind('page_count', update_counts)
        fbind('up_count', update_counts)
        fbind('right_count', update_counts)
        fbind('scroll_count', update_counts)

    def select_with_touch(self, node, touch=None):
        """(internal) Processes a touch on the node. This should be called by
        the derived widget when a node is touched and is to be used for
        selection. Depending on the keyboard keys pressed and the
        configuration, it could select or deslect this and other nodes in the
        selectable nodes list, :meth:`get_selectable_nodes`.

        :Parameters:
            `node`
                The node that received the touch. Can be None for a scroll
                type touch.
            `touch`
                Optionally, the touch. Defaults to None.

        :Returns:
            bool, True if the touch was used, False otherwise.
        """
        multi = self.multiselect
        multiselect = multi and (self._ctrl_down or self.touch_multiselect)
        range_select = multi and self._shift_down
        if touch and 'button' in touch.profile and (touch.button in ('scrollup', 'scrolldown', 'scrollleft', 'scrollright')):
            node_src, idx_src = self._resolve_last_node()
            node, idx = self.goto_node(touch.button, node_src, idx_src)
            if node == node_src:
                return False
            if range_select:
                self._select_range(multiselect, True, node, idx)
            else:
                if not multiselect:
                    self.clear_selection()
                self.select_node(node)
            return True
        if node is None:
            return False
        if node in self.selected_nodes and (not range_select):
            if multiselect:
                self.deselect_node(node)
            else:
                selected_node_count = len(self.selected_nodes)
                self.clear_selection()
                if not self.touch_deselect_last or selected_node_count > 1:
                    self.select_node(node)
        elif range_select:
            self._select_range(multiselect, not multiselect, node, 0)
        else:
            if not multiselect:
                self.clear_selection()
            self.select_node(node)
        return True

    def select_with_key_down(self, keyboard, scancode, codepoint, modifiers, **kwargs):
        """Processes a key press. This is called when a key press is to be used
        for selection. Depending on the keyboard keys pressed and the
        configuration, it could select or deselect nodes or node ranges
        from the selectable nodes list, :meth:`get_selectable_nodes`.

        The parameters are such that it could be bound directly to the
        on_key_down event of a keyboard. Therefore, it is safe to be called
        repeatedly when the key is held down as is done by the keyboard.

        :Returns:
            bool, True if the keypress was used, False otherwise.
        """
        if not self.keyboard_select:
            return False
        keys = self._key_list
        multi = self.multiselect
        node_src, idx_src = self._resolve_last_node()
        text = scancode[1]
        if text == 'shift':
            self._shift_down = True
        elif text in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = True
        elif multi and 'ctrl' in modifiers and (text in ('a', 'A')) and (text not in keys):
            sister_nodes = self.get_selectable_nodes()
            select = self.select_node
            for node in sister_nodes:
                select(node)
            keys.append(text)
        else:
            s = text
            if len(text) > 1:
                d = {'divide': '/', 'mul': '*', 'substract': '-', 'add': '+', 'decimal': '.'}
                if text.startswith('numpad'):
                    s = text[6:]
                    if len(s) > 1:
                        if s in d:
                            s = d[s]
                        else:
                            s = None
                else:
                    s = None
            if s is not None:
                if s not in keys:
                    if time() - self._last_key_time <= self.text_entry_timeout:
                        self._word_filter += s
                    else:
                        self._word_filter = s
                    keys.append(s)
                self._last_key_time = time()
                node, idx = self.goto_node(self._word_filter, node_src, idx_src)
            else:
                self._word_filter = ''
                node, idx = self.goto_node(text, node_src, idx_src)
            if node == node_src:
                return False
            multiselect = multi and 'ctrl' in modifiers
            if multi and 'shift' in modifiers:
                self._select_range(multiselect, True, node, idx)
            else:
                if not multiselect:
                    self.clear_selection()
                self.select_node(node)
            return True
        self._word_filter = ''
        return False

    def select_with_key_up(self, keyboard, scancode, **kwargs):
        """(internal) Processes a key release. This must be called by the
        derived widget when a key that :meth:`select_with_key_down` returned
        True is released.

        The parameters are such that it could be bound directly to the
        on_key_up event of a keyboard.

        :Returns:
            bool, True if the key release was used, False otherwise.
        """
        if scancode[1] == 'shift':
            self._shift_down = False
        elif scancode[1] in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = False
        else:
            try:
                self._key_list.remove(scancode[1])
                return True
            except ValueError:
                return False
        return True

    def _update_counts(self, *largs):
        pc = self.page_count
        uc = self.up_count
        rc = self.right_count
        sc = self.scroll_count
        self._offset_counts = {'pageup': -pc, 'pagedown': pc, 'up': -uc, 'down': uc, 'right': rc, 'left': -rc, 'scrollup': sc, 'scrolldown': -sc, 'scrollright': -sc, 'scrollleft': sc}

    def _resolve_last_node(self):
        sister_nodes = self.get_selectable_nodes()
        if not len(sister_nodes):
            return (None, 0)
        last_node = self._last_selected_node
        last_idx = self._last_node_idx
        end = len(sister_nodes) - 1
        if last_node is None:
            last_node = self._anchor
            last_idx = self._anchor_idx
        if last_node is None:
            return (sister_nodes[end], end)
        if last_idx > end or sister_nodes[last_idx] != last_node:
            try:
                return (last_node, self.get_index_of_node(last_node, sister_nodes))
            except ValueError:
                return (sister_nodes[end], end)
        return (last_node, last_idx)

    def _select_range(self, multiselect, keep_anchor, node, idx):
        """Selects a range between self._anchor and node or idx.
        If multiselect is True, it will be added to the selection, otherwise
        it will unselect everything before selecting the range. This is only
        called if self.multiselect is True.
        If keep anchor is False, the anchor is moved to node. This should
        always be True for keyboard selection.
        """
        select = self.select_node
        sister_nodes = self.get_selectable_nodes()
        end = len(sister_nodes) - 1
        last_node = self._anchor
        last_idx = self._anchor_idx
        if last_node is None:
            last_idx = end
            last_node = sister_nodes[end]
        elif last_idx > end or sister_nodes[last_idx] != last_node:
            try:
                last_idx = self.get_index_of_node(last_node, sister_nodes)
            except ValueError:
                return
        if idx > end or sister_nodes[idx] != node:
            try:
                idx = self.get_index_of_node(node, sister_nodes)
            except ValueError:
                return
        if last_idx > idx:
            last_idx, idx = (idx, last_idx)
        if not multiselect:
            self.clear_selection()
        for item in sister_nodes[last_idx:idx + 1]:
            select(item)
        if keep_anchor:
            self._anchor = last_node
            self._anchor_idx = last_idx
        else:
            self._anchor = node
            self._anchor_idx = idx
        self._last_selected_node = node
        self._last_node_idx = idx

    def clear_selection(self):
        """ Deselects all the currently selected nodes.
        """
        deselect = self.deselect_node
        nodes = self.selected_nodes
        for node in nodes[:]:
            deselect(node)

    def get_selectable_nodes(self):
        """(internal) Returns a list of the nodes that can be selected. It can
        be overwritten by the derived widget to return the correct list.

        This list is used to determine which nodes to select with group
        selection. E.g. the last element in the list will be selected when
        home is pressed, pagedown will move (or add to, if shift is held) the
        selection from the current position by negative :attr:`page_count`
        nodes starting from the position of the currently selected node in
        this list and so on. Still, nodes can be selected even if they are not
        in this list.

        .. note::

            It is safe to dynamically change this list including removing,
            adding, or re-arranging its elements. Nodes can be selected even
            if they are not on this list. And selected nodes removed from the
            list will remain selected until :meth:`deselect_node` is called.

        .. warning::

            Layouts display their children in the reverse order. That is, the
            contents of :attr:`~kivy.uix.widget.Widget.children` is displayed
            form right to left, bottom to top. Therefore, internally, the
            indices of the elements returned by this function are reversed to
            make it work by default for most layouts so that the final result
            is consistent e.g. home, although it will select the last element
            in this list visually, will select the first element when
            counting from top to bottom and left to right. If this behavior is
            not desired, a reversed list should be returned instead.

        Defaults to returning :attr:`~kivy.uix.widget.Widget.children`.
        """
        return self.children

    def get_index_of_node(self, node, selectable_nodes):
        """(internal) Returns the index of the `node` within the
        `selectable_nodes` returned by :meth:`get_selectable_nodes`.
        """
        return selectable_nodes.index(node)

    def goto_node(self, key, last_node, last_node_idx):
        """(internal) Used by the controller to get the node at the position
        indicated by key. The key can be keyboard inputs, e.g. pageup,
        or scroll inputs from the mouse scroll wheel, e.g. scrollup.
        'last_node' is the last node selected and is used to find the resulting
        node. For example, if the key is up, the returned node is one node
        up from the last node.

        It can be overwritten by the derived widget.

        :Parameters:
            `key`
                str, the string used to find the desired node. It can be any
                of the keyboard keys, as well as the mouse scrollup,
                scrolldown, scrollright, and scrollleft strings. If letters
                are typed in quick succession, the letters will be combined
                before it's passed in as key and can be used to find nodes that
                have an associated string that starts with those letters.
            `last_node`
                The last node that was selected.
            `last_node_idx`
                The cached index of the last node selected in the
                :meth:`get_selectable_nodes` list. If the list hasn't changed
                it saves having to look up the index of `last_node` in that
                list.

        :Returns:
            tuple, the node targeted by key and its index in the
            :meth:`get_selectable_nodes` list. Returning
            `(last_node, last_node_idx)` indicates a node wasn't found.
        """
        sister_nodes = self.get_selectable_nodes()
        end = len(sister_nodes) - 1
        counts = self._offset_counts
        if end == -1:
            return (last_node, last_node_idx)
        if last_node_idx > end or sister_nodes[last_node_idx] != last_node:
            try:
                last_node_idx = self.get_index_of_node(last_node, sister_nodes)
            except ValueError:
                return (last_node, last_node_idx)
        is_reversed = self.nodes_order_reversed
        if key in counts:
            count = -counts[key] if is_reversed else counts[key]
            idx = max(min(count + last_node_idx, end), 0)
            return (sister_nodes[idx], idx)
        elif key == 'home':
            if is_reversed:
                return (sister_nodes[end], end)
            return (sister_nodes[0], 0)
        elif key == 'end':
            if is_reversed:
                return (sister_nodes[0], 0)
            return (sister_nodes[end], end)
        else:
            return (last_node, last_node_idx)

    def select_node(self, node):
        """ Selects a node.

        It is called by the controller when it selects a node and can be
        called from the outside to select a node directly. The derived widget
        should overwrite this method and change the node state to selected
        when called.

        :Parameters:
            `node`
                The node to be selected.

        :Returns:
            bool, True if the node was selected, False otherwise.

        .. warning::

            This method must be called by the derived widget using super if it
            is overwritten.
        """
        nodes = self.selected_nodes
        if node in nodes:
            return False
        if not self.multiselect and len(nodes):
            self.clear_selection()
        if node not in nodes:
            nodes.append(node)
        self._anchor = node
        self._last_selected_node = node
        return True

    def deselect_node(self, node):
        """ Deselects a possibly selected node.

        It is called by the controller when it deselects a node and can also
        be called from the outside to deselect a node directly. The derived
        widget should overwrite this method and change the node to its
        unselected state when this is called

        :Parameters:
            `node`
                The node to be deselected.

        .. warning::

            This method must be called by the derived widget using super if it
            is overwritten.
        """
        try:
            self.selected_nodes.remove(node)
            return True
        except ValueError:
            return False
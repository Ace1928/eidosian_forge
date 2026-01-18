from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
class LayoutSelectionBehavior(CompoundSelectionBehavior):
    """The :class:`LayoutSelectionBehavior` can be combined with
    :class:`RecycleLayoutManagerBehavior` to allow its derived classes
    selection behaviors similarly to how
    :class:`~kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`
    can be used to add selection behaviors to normal layout.

    :class:`RecycleLayoutManagerBehavior` manages its children
    differently than normal layouts or widgets so this class adapts
    :class:`~kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`
    based selection to work with :class:`RecycleLayoutManagerBehavior` as well.

    Similarly to
    :class:`~kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`,
    one can select using the keyboard or touch, which calls :meth:`select_node`
    or :meth:`deselect_node`, or one can call these methods directly. When a
    item is selected or deselected :meth:`apply_selection` is called. See
    :meth:`apply_selection`.


    """
    key_selection = StringProperty(None, allownone=True)
    'The key used to check whether a view of a data item can be selected\n    with touch or the keyboard.\n\n    :attr:`key_selection` is the key in data, which if present and ``True``\n    will enable selection for this item from the keyboard or with a touch.\n    When None, the default, not item will be selectable.\n\n    :attr:`key_selection` is a :class:`StringProperty` and defaults to None.\n\n    .. note::\n        All data items can be selected directly using :meth:`select_node` or\n        :meth:`deselect_node`, even if :attr:`key_selection` is False.\n    '
    _selectable_nodes = []
    _nodes_map = {}

    def __init__(self, **kwargs):
        self.nodes_order_reversed = False
        super(LayoutSelectionBehavior, self).__init__(**kwargs)

    def compute_sizes_from_data(self, data, flags):
        key = self.key_selection
        if key is None:
            nodes = self._selectable_nodes = []
        else:
            nodes = self._selectable_nodes = [i for i, d in enumerate(data) if d.get(key)]
        self._nodes_map = {v: k for k, v in enumerate(nodes)}
        return super(LayoutSelectionBehavior, self).compute_sizes_from_data(data, flags)

    def get_selectable_nodes(self):
        return self._selectable_nodes

    def get_index_of_node(self, node, selectable_nodes):
        return self._nodes_map[node]

    def goto_node(self, key, last_node, last_node_idx):
        node, idx = super(LayoutSelectionBehavior, self).goto_node(key, last_node, last_node_idx)
        if node is not last_node:
            self.goto_view(node)
        return (node, idx)

    def select_node(self, node):
        if super(LayoutSelectionBehavior, self).select_node(node):
            view = self.recycleview.view_adapter.get_visible_view(node)
            if view is not None:
                self.apply_selection(node, view, True)

    def deselect_node(self, node):
        if super(LayoutSelectionBehavior, self).deselect_node(node):
            view = self.recycleview.view_adapter.get_visible_view(node)
            if view is not None:
                self.apply_selection(node, view, False)

    def apply_selection(self, index, view, is_selected):
        """Applies the selection to the view. This is called internally when
        a view is displayed and it needs to be shown as selected or as not
        selected.

        It is called when :meth:`select_node` or :meth:`deselect_node` is
        called or when a view needs to be refreshed. Its function is purely to
        update the view to reflect the selection state. So the function may be
        called multiple times even if the selection state may not have changed.

        If the view is a instance of
        :class:`~kivy.uix.recycleview.views.RecycleDataViewBehavior`, its
        :meth:`~kivy.uix.recycleview.views.RecycleDataViewBehavior.apply_selection` method will be called every time the view needs to refresh
        the selection state. Otherwise, the this method is responsible
        for applying the selection.

        :Parameters:

            `index`: int
                The index of the data item that is associated with the view.
            `view`: widget
                The widget that is the view of this data item.
            `is_selected`: bool
                Whether the item is selected.
        """
        viewclass = view.__class__
        if viewclass not in _view_base_cache:
            _view_base_cache[viewclass] = isinstance(view, RecycleDataViewBehavior)
        if _view_base_cache[viewclass]:
            view.apply_selection(self.recycleview, index, is_selected)

    def refresh_view_layout(self, index, layout, view, viewport):
        super(LayoutSelectionBehavior, self).refresh_view_layout(index, layout, view, viewport)
        self.apply_selection(index, view, index in self.selected_nodes)
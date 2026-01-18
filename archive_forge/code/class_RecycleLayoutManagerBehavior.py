from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
class RecycleLayoutManagerBehavior(object):
    """A RecycleLayoutManagerBehavior is responsible for positioning views into
    the :attr:`RecycleView.data` within a :class:`RecycleView`. It adds new
    views into the data when it becomes visible to the user, and removes them
    when they leave the visible area.
    """
    viewclass = ObjectProperty(None)
    'See :attr:`RecyclerView.viewclass`.\n    '
    key_viewclass = StringProperty(None)
    'See :attr:`RecyclerView.key_viewclass`.\n    '
    recycleview = ObjectProperty(None, allownone=True)
    asked_sizes = None

    def attach_recycleview(self, rv):
        self.recycleview = rv
        if rv:
            fbind = self.fbind
            fbind('viewclass', rv.refresh_from_data)
            fbind('key_viewclass', rv.refresh_from_data)
            fbind('viewclass', rv._dispatch_prop_on_source, 'viewclass')
            fbind('key_viewclass', rv._dispatch_prop_on_source, 'key_viewclass')

    def detach_recycleview(self):
        self.clear_layout()
        rv = self.recycleview
        if rv:
            funbind = self.funbind
            funbind('viewclass', rv.refresh_from_data)
            funbind('key_viewclass', rv.refresh_from_data)
            funbind('viewclass', rv._dispatch_prop_on_source, 'viewclass')
            funbind('key_viewclass', rv._dispatch_prop_on_source, 'key_viewclass')
        self.recycleview = None

    def compute_sizes_from_data(self, data, flags):
        pass

    def compute_layout(self, data, flags):
        pass

    def compute_visible_views(self, data, viewport):
        """`viewport` is in coordinates of the layout manager.
        """
        pass

    def set_visible_views(self, indices, data, viewport):
        """`viewport` is in coordinates of the layout manager.
        """
        pass

    def refresh_view_layout(self, index, layout, view, viewport):
        """`See :meth:`~kivy.uix.recycleview.views.RecycleDataAdapter.refresh_view_layout`.
        """
        self.recycleview.view_adapter.refresh_view_layout(index, layout, view, viewport)

    def get_view_index_at(self, pos):
        """Return the view `index` on which position, `pos`, falls.

        `pos` is in coordinates of the layout manager.
        """
        pass

    def remove_views(self):
        rv = self.recycleview
        if rv:
            adapter = rv.view_adapter
            if adapter:
                adapter.make_views_dirty()

    def remove_view(self, view, index):
        rv = self.recycleview
        if rv:
            adapter = rv.view_adapter
            if adapter:
                adapter.make_view_dirty(view, index)

    def clear_layout(self):
        rv = self.recycleview
        if rv:
            adapter = rv.view_adapter
            if adapter:
                adapter.invalidate()

    def goto_view(self, index):
        """Moves the views so that the view corresponding to `index` is
        visible.
        """
        pass

    def on_viewclass(self, instance, value):
        if isinstance(value, string_types):
            self.viewclass = getattr(Factory, value)
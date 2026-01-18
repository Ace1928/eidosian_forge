from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
class RecycleView(RecycleViewBehavior, ScrollView):
    """
    RecycleView is a flexible view for providing a limited window
    into a large data set.

    See the module documentation for more information.
    """

    def __init__(self, **kwargs):
        if self.data_model is None:
            kwargs.setdefault('data_model', RecycleDataModel())
        if self.view_adapter is None:
            kwargs.setdefault('view_adapter', RecycleDataAdapter())
        super(RecycleView, self).__init__(**kwargs)
        fbind = self.fbind
        fbind('scroll_x', self.refresh_from_viewport)
        fbind('scroll_y', self.refresh_from_viewport)
        fbind('size', self.refresh_from_viewport)
        self.refresh_from_data()

    def _convert_sv_to_lm(self, x, y):
        lm = self.layout_manager
        tree = [lm]
        parent = lm.parent
        while parent is not None and parent is not self:
            tree.append(parent)
            parent = parent.parent
        if parent is not self:
            raise Exception('The layout manager must be a sub child of the recycleview. Could not find {} in the parent tree of {}'.format(self, lm))
        for widget in reversed(tree):
            x, y = widget.to_local(x, y)
        return (x, y)

    def get_viewport(self):
        lm = self.layout_manager
        lm_w, lm_h = lm.size
        w, h = self.size
        scroll_y = min(1, max(self.scroll_y, 0))
        scroll_x = min(1, max(self.scroll_x, 0))
        if lm_h <= h:
            bottom = 0
        else:
            above = (lm_h - h) * scroll_y
            bottom = max(0, lm_h - above - h)
        bottom = max(0, (lm_h - h) * scroll_y)
        left = max(0, (lm_w - w) * scroll_x)
        width = min(w, lm_w)
        height = min(h, lm_h)
        left, bottom = self._convert_sv_to_lm(left, bottom)
        return (left, bottom, width, height)

    def save_viewport(self):
        pass

    def restore_viewport(self):
        pass

    def add_widget(self, widget, *args, **kwargs):
        super(RecycleView, self).add_widget(widget, *args, **kwargs)
        if isinstance(widget, RecycleLayoutManagerBehavior) and (not self.layout_manager):
            self.layout_manager = widget

    def remove_widget(self, widget, *args, **kwargs):
        super(RecycleView, self).remove_widget(widget, *args, **kwargs)
        if self.layout_manager == widget:
            self.layout_manager = None

    def _get_data(self):
        d = self.data_model
        return d and d.data

    def _set_data(self, value):
        d = self.data_model
        if d is not None:
            d.data = value
    data = AliasProperty(_get_data, _set_data, bind=['data_model'])
    '\n    The data used by the current view adapter. This is a list of dicts whose\n    keys map to the corresponding property names of the\n    :attr:`~RecycleView.viewclass`.\n\n    data is an :class:`~kivy.properties.AliasProperty` that gets and sets the\n    data used to generate the views.\n    '

    def _get_viewclass(self):
        a = self.layout_manager
        return a and a.viewclass

    def _set_viewclass(self, value):
        a = self.layout_manager
        if a:
            a.viewclass = value
    viewclass = AliasProperty(_get_viewclass, _set_viewclass, bind=['layout_manager'])
    '\n    The viewclass used by the current layout_manager.\n\n    viewclass is an :class:`~kivy.properties.AliasProperty` that gets and sets\n    the class used to generate the individual items presented in the view.\n    '

    def _get_key_viewclass(self):
        a = self.layout_manager
        return a and a.key_viewclass

    def _set_key_viewclass(self, value):
        a = self.layout_manager
        if a:
            a.key_viewclass = value
    key_viewclass = AliasProperty(_get_key_viewclass, _set_key_viewclass, bind=['layout_manager'])
    '\n    key_viewclass is an :class:`~kivy.properties.AliasProperty` that gets and\n    sets the key viewclass for the current\n    :attr:`~kivy.uix.recycleview.layout_manager`.\n    '
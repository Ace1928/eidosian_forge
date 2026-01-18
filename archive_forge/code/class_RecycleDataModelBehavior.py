from kivy.properties import ListProperty, ObservableDict, ObjectProperty
from kivy.event import EventDispatcher
from functools import partial
class RecycleDataModelBehavior(object):
    """:class:`RecycleDataModelBehavior` is the base class for the models
    that describes and provides the data for the
    :class:`~kivy.uix.recycleview.RecycleViewBehavior`.

    :Events:
        `on_data_changed`:
            Fired when the data changes. The event may dispatch
            keyword arguments specific to each implementation of the data
            model.
            When dispatched, the event and keyword arguments are forwarded to
            :meth:`~kivy.uix.recycleview.RecycleViewBehavior.refresh_from_data`.
    """
    __events__ = ('on_data_changed',)
    recycleview = ObjectProperty(None, allownone=True)
    'The\n    :class:`~kivy.uix.recycleview.RecycleViewBehavior` instance\n    associated with this data model.\n    '

    def attach_recycleview(self, rv):
        """Associates a
        :class:`~kivy.uix.recycleview.RecycleViewBehavior` with
        this data model.
        """
        self.recycleview = rv
        if rv:
            self.fbind('on_data_changed', rv.refresh_from_data)

    def detach_recycleview(self):
        """Removes the
        :class:`~kivy.uix.recycleview.RecycleViewBehavior`
        associated with this data model.
        """
        rv = self.recycleview
        if rv:
            self.funbind('on_data_changed', rv.refresh_from_data)
        self.recycleview = None

    def on_data_changed(self, *largs, **kwargs):
        pass
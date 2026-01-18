from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
class RecycleViewBehavior(object):
    """RecycleViewBehavior provides a behavioral model upon which the
    :class:`RecycleView` is built. Together, they offer an extensible and
    flexible way to produce views with limited windows over large data sets.

    See the module documentation for more information.
    """
    _view_adapter = None
    _data_model = None
    _layout_manager = None
    _refresh_flags = {'data': [], 'layout': [], 'viewport': False}
    _refresh_trigger = None

    def __init__(self, **kwargs):
        self._refresh_trigger = Clock.create_trigger(self.refresh_views, -1)
        self._refresh_flags = deepcopy(self._refresh_flags)
        super(RecycleViewBehavior, self).__init__(**kwargs)

    def get_viewport(self):
        pass

    def save_viewport(self):
        pass

    def restore_viewport(self):
        pass

    def refresh_views(self, *largs):
        lm = self.layout_manager
        flags = self._refresh_flags
        if lm is None or self.view_adapter is None or self.data_model is None:
            return
        data = self.data
        f = flags['data']
        if f:
            self.save_viewport()
            flags['data'] = []
            flags['layout'] = [{}]
            lm.compute_sizes_from_data(data, f)
        while flags['layout']:
            self.save_viewport()
            if flags['data']:
                return
            flags['viewport'] = True
            f = flags['layout']
            flags['layout'] = []
            try:
                lm.compute_layout(data, f)
            except LayoutChangeException:
                flags['layout'].append({})
                continue
        if flags['data']:
            return
        self._refresh_trigger.cancel()
        self.restore_viewport()
        if flags['viewport']:
            flags['viewport'] = False
            viewport = self.get_viewport()
            indices = lm.compute_visible_views(data, viewport)
            lm.set_visible_views(indices, data, viewport)

    def refresh_from_data(self, *largs, **kwargs):
        """
        This should be called when data changes. Data changes typically
        indicate that everything should be recomputed since the source data
        changed.

        This method is automatically bound to the
        :attr:`~RecycleDataModelBehavior.on_data_changed` method of the
        :class:`~RecycleDataModelBehavior` class and
        therefore responds to and accepts the keyword arguments of that event.

        It can be called manually to trigger an update.
        """
        self._refresh_flags['data'].append(kwargs)
        self._refresh_trigger()

    def refresh_from_layout(self, *largs, **kwargs):
        """
        This should be called when the layout changes or needs to change. It is
        typically called when a layout parameter has changed and therefore the
        layout needs to be recomputed.
        """
        self._refresh_flags['layout'].append(kwargs)
        self._refresh_trigger()

    def refresh_from_viewport(self, *largs):
        """
        This should be called when the viewport changes and the displayed data
        must be updated. Neither the data nor the layout will be recomputed.
        """
        self._refresh_flags['viewport'] = True
        self._refresh_trigger()

    def _dispatch_prop_on_source(self, prop_name, *largs):
        getattr(self.__class__, prop_name).dispatch(self)

    def _get_data_model(self):
        return self._data_model

    def _set_data_model(self, value):
        data_model = self._data_model
        if value is data_model:
            return
        if data_model is not None:
            self._data_model = None
            data_model.detach_recycleview()
        if value is None:
            return True
        if not isinstance(value, RecycleDataModelBehavior):
            raise ValueError('Expected object based on RecycleDataModelBehavior, got {}'.format(value.__class__))
        self._data_model = value
        value.attach_recycleview(self)
        self.refresh_from_data()
        return True
    data_model = AliasProperty(_get_data_model, _set_data_model)
    '\n    The Data model responsible for maintaining the data set.\n\n    data_model is an :class:`~kivy.properties.AliasProperty` that gets and sets\n    the current data model.\n    '

    def _get_view_adapter(self):
        return self._view_adapter

    def _set_view_adapter(self, value):
        view_adapter = self._view_adapter
        if value is view_adapter:
            return
        if view_adapter is not None:
            self._view_adapter = None
            view_adapter.detach_recycleview()
        if value is None:
            return True
        if not isinstance(value, RecycleDataAdapter):
            raise ValueError('Expected object based on RecycleAdapter, got {}'.format(value.__class__))
        self._view_adapter = value
        value.attach_recycleview(self)
        self.refresh_from_layout()
        return True
    view_adapter = AliasProperty(_get_view_adapter, _set_view_adapter)
    '\n    The adapter responsible for providing views that represent items in a data\n    set.\n\n    view_adapter is an :class:`~kivy.properties.AliasProperty` that gets and\n    sets the current view adapter.\n    '

    def _get_layout_manager(self):
        return self._layout_manager

    def _set_layout_manager(self, value):
        lm = self._layout_manager
        if value is lm:
            return
        if lm is not None:
            self._layout_manager = None
            lm.detach_recycleview()
        if value is None:
            return True
        if not isinstance(value, RecycleLayoutManagerBehavior):
            raise ValueError('Expected object based on RecycleLayoutManagerBehavior, got {}'.format(value.__class__))
        self._layout_manager = value
        value.attach_recycleview(self)
        self.refresh_from_layout()
        return True
    layout_manager = AliasProperty(_get_layout_manager, _set_layout_manager)
    '\n    The Layout manager responsible for positioning views within the\n    :class:`RecycleView`.\n\n    layout_manager is an :class:`~kivy.properties.AliasProperty` that gets\n    and sets the layout_manger.\n    '
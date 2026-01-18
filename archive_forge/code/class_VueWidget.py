from traitlets import Unicode, Instance, Union, List, Any, Dict
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget_layout import Layout
from ipywidgets.widgets.widget import widget_serialization, CallbackDispatcher
from ipywidgets.widgets.trait_types import InstanceDict
from ._version import semver
from .ForceLoad import force_load_instance
class VueWidget(DOMWidget, Events):
    layout = InstanceDict(Layout, allow_none=True).tag(sync=True, **widget_serialization)
    _jupyter_vue = Any(force_load_instance, read_only=True).tag(sync=True, **widget_serialization)
    _model_name = Unicode('VueModel').tag(sync=True)
    _view_name = Unicode('VueView').tag(sync=True)
    _view_module = Unicode('jupyter-vue').tag(sync=True)
    _model_module = Unicode('jupyter-vue').tag(sync=True)
    _view_module_version = Unicode(semver).tag(sync=True)
    _model_module_version = Unicode(semver).tag(sync=True)
    children = List(Union([Instance(DOMWidget), Unicode()])).tag(sync=True, **widget_serialization)
    slot = Unicode(None, allow_none=True).tag(sync=True)
    _events = List(Unicode()).tag(sync=True)
    v_model = Any('!!disabled!!', allow_none=True).tag(sync=True)
    style_ = Unicode(None, allow_none=True).tag(sync=True)
    class_ = Unicode(None, allow_none=True).tag(sync=True)
    attributes = Dict(None, allow_none=True).tag(sync=True)
    v_slots = List(Dict()).tag(sync=True, **widget_serialization)
    v_on = Unicode(None, allow_none=True).tag(sync=True)

    def __init__(self, **kwargs):
        self.class_list = ClassList(self)
        super().__init__(**kwargs)

    def show(self):
        """Make the widget visible"""
        self.class_list.remove('d-none')

    def hide(self):
        """Make the widget invisible"""
        self.class_list.add('d-none')
from __future__ import absolute_import
import ipywidgets as widgets
from bokeh.models import CustomJS
from traitlets import Unicode
import ipyvolume
@widgets.register
class WidgetManagerHackModel(widgets.Widget):
    _model_name = Unicode('WidgetManagerHackModel').tag(sync=True)
    _model_module = Unicode('ipyvolume').tag(sync=True)
    _model_module_version = Unicode(semver_range_frontend).tag(sync=True)
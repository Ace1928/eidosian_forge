from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ExpansionPanel(VuetifyWidget):
    _model_name = Unicode('ExpansionPanelModel').tag(sync=True)
    active_class = Unicode(None, allow_none=True).tag(sync=True)
    disabled = Bool(None, allow_none=True).tag(sync=True)
    readonly = Bool(None, allow_none=True).tag(sync=True)
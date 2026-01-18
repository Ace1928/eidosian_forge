from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ExpandTransition(VuetifyWidget):
    _model_name = Unicode('ExpandTransitionModel').tag(sync=True)
    mode = Unicode(None, allow_none=True).tag(sync=True)
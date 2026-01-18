from traitlets import (
from .VuetifyWidget import VuetifyWidget
class StepperContent(VuetifyWidget):
    _model_name = Unicode('StepperContentModel').tag(sync=True)
    step = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
from traitlets import (
from .VuetifyWidget import VuetifyWidget
class Stepper(VuetifyWidget):
    _model_name = Unicode('StepperModel').tag(sync=True)
    alt_labels = Bool(None, allow_none=True).tag(sync=True)
    dark = Bool(None, allow_none=True).tag(sync=True)
    light = Bool(None, allow_none=True).tag(sync=True)
    non_linear = Bool(None, allow_none=True).tag(sync=True)
    value = Any(None, allow_none=True).tag(sync=True)
    vertical = Bool(None, allow_none=True).tag(sync=True)
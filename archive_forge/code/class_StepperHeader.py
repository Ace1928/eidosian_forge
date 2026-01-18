from traitlets import (
from .VuetifyWidget import VuetifyWidget
class StepperHeader(VuetifyWidget):
    _model_name = Unicode('StepperHeaderModel').tag(sync=True)
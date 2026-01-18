from traitlets import (
from .VuetifyWidget import VuetifyWidget
class TableOverflow(VuetifyWidget):
    _model_name = Unicode('TableOverflowModel').tag(sync=True)
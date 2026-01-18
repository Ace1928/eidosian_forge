from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ListItemContent(VuetifyWidget):
    _model_name = Unicode('ListItemContentModel').tag(sync=True)
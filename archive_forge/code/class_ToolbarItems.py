from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ToolbarItems(VuetifyWidget):
    _model_name = Unicode('ToolbarItemsModel').tag(sync=True)
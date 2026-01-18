from traitlets import (
from .VuetifyWidget import VuetifyWidget
class CardText(VuetifyWidget):
    _model_name = Unicode('CardTextModel').tag(sync=True)
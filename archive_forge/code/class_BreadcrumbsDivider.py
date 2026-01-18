from traitlets import (
from .VuetifyWidget import VuetifyWidget
class BreadcrumbsDivider(VuetifyWidget):
    _model_name = Unicode('BreadcrumbsDividerModel').tag(sync=True)
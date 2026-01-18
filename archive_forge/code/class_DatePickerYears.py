from traitlets import (
from .VuetifyWidget import VuetifyWidget
class DatePickerYears(VuetifyWidget):
    _model_name = Unicode('DatePickerYearsModel').tag(sync=True)
    color = Unicode(None, allow_none=True).tag(sync=True)
    locale = Unicode(None, allow_none=True).tag(sync=True)
    max = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    min = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    readonly = Bool(None, allow_none=True).tag(sync=True)
    value = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
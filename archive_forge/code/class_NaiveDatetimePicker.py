from traitlets import Unicode, Bool, validate, TraitError
from .trait_types import datetime_serialization, Datetime, naive_serialization
from .valuewidget import ValueWidget
from .widget import register
from .widget_core import CoreWidget
from .widget_description import DescriptionWidget
@register
class NaiveDatetimePicker(DatetimePicker):
    """
    Display a widget for picking naive datetimes (i.e. timezone unaware).

    Parameters
    ----------

    value: datetime.datetime
        The current value of the widget.

    disabled: bool
        Whether to disable user changes.

    min: datetime.datetime
        The lower allowed datetime bound

    max: datetime.datetime
        The upper allowed datetime bound

    Examples
    --------

    >>> import datetime
    >>> import ipydatetime
    >>> datetime_pick = ipydatetime.NaiveDatetimePicker()
    >>> datetime_pick.value = datetime.datetime(2018, 09, 5, 12, 34, 3)
    """
    _model_name = Unicode('NaiveDatetimeModel').tag(sync=True)
    value = Datetime(None, allow_none=True).tag(sync=True, **naive_serialization)
    min = Datetime(None, allow_none=True).tag(sync=True, **naive_serialization)
    max = Datetime(None, allow_none=True).tag(sync=True, **naive_serialization)

    def _validate_tz(self, value):
        if value.tzinfo is not None:
            raise TraitError('%s values needs to be timezone unaware' % (self.__class__.__name__,))
        return value
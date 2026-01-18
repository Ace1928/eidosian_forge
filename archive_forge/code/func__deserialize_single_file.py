import datetime as dt
from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_button import ButtonStyle
from .widget import register, widget_serialization
from .trait_types import InstanceDict, TypedTuple
from traitlets import Bunch
def _deserialize_single_file(js):
    uploaded_file = Bunch()
    for attribute in ['name', 'type', 'size', 'content']:
        uploaded_file[attribute] = js[attribute]
    uploaded_file['last_modified'] = dt.datetime.fromtimestamp(js['last_modified'] / 1000, tz=dt.timezone.utc)
    return uploaded_file
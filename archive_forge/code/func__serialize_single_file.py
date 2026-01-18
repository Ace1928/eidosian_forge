import datetime as dt
from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_button import ButtonStyle
from .widget import register, widget_serialization
from .trait_types import InstanceDict, TypedTuple
from traitlets import Bunch
def _serialize_single_file(uploaded_file):
    js = {}
    for attribute in ['name', 'type', 'size', 'content']:
        js[attribute] = uploaded_file[attribute]
    js['last_modified'] = int(uploaded_file['last_modified'].timestamp() * 1000)
    return js
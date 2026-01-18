import uuid
import numpy as np
from ..data_utils import is_pandas_df, has_geo_interface, records_from_geo_interface
from .json_tools import JSONMixin, camel_and_lower
from ..settings import settings as pydeck_settings
from pydeck.types import Image, Function
from pydeck.exceptions import BinaryTransportException
def _add_default_layer_attributes(self, kwargs):
    attributes = pydeck_settings.default_layer_attributes
    if isinstance(attributes, dict) and self.type in attributes and isinstance(attributes[self.type], dict):
        kwargs = {**attributes[self.type], **kwargs}
    return kwargs
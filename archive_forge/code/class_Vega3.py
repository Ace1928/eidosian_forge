from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
class Vega3(Panel):

    @property
    def view_type(self) -> str:
        return 'Vega3'
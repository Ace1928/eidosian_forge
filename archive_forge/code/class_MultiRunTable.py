from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
class MultiRunTable(Panel):

    @property
    def view_type(self) -> str:
        return 'Multi Run Table'
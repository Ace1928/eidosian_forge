from __future__ import annotations
from base64 import b64encode
from pathlib import Path
from typing import (
import param
from param.parameterized import eval_function_with_deps, iscoroutinefunction
from pyviz_comms import JupyterComm
from ..io.notebook import push
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import (
from ..util import lazy_load
from .base import Widget
from .button import BUTTON_STYLES, BUTTON_TYPES, IconMixin
from .indicators import Progress  # noqa
class JSONEditor(Widget):
    """
    The `JSONEditor` provides a visual editor for JSON-serializable
    datastructures, e.g. Python dictionaries and lists, with functionality for
    different editing modes, inserting objects and validation using JSON
    Schema.

    Reference: https://panel.holoviz.org/reference/widgets/JSONEditor.html

    :Example:

    >>> JSONEditor(value={
    ...     'dict'  : {'key': 'value'},
    ...     'float' : 3.14,
    ...     'int'   : 1,
    ...     'list'  : [1, 2, 3],
    ...     'string': 'A string',
    ... }, mode='code')
    """
    menu = param.Boolean(default=True, doc='\n        Adds main menu bar - Contains format, sort, transform, search\n        etc. functionality. true by default. Applicable in all types\n        of mode.')
    mode = param.Selector(default='tree', objects=['tree', 'view', 'form', 'text', 'preview'], doc="\n        Sets the editor mode. In 'view' mode, the data and\n        datastructure is read-only. In 'form' mode, only the value can\n        be changed, the data structure is read-only. Mode 'code'\n        requires the Ace editor to be loaded on the page. Mode 'text'\n        shows the data as plain text. The 'preview' mode can handle\n        large JSON documents up to 500 MiB. It shows a preview of the\n        data, and allows to transform, sort, filter, format, or\n        compact the data.")
    search = param.Boolean(default=True, doc="\n        Enables a search box in the upper right corner of the\n        JSONEditor. true by default. Only applicable when mode is\n        'tree', 'view', or 'form'.")
    selection = param.List(default=[], doc='\n        Current selection.')
    schema = param.Dict(default=None, doc='\n        Validate the JSON object against a JSON schema. A JSON schema\n        describes the structure that a JSON object must have, like\n        required properties or the type that a value must have.\n\n        See http://json-schema.org/ for more information.')
    templates = param.List(doc='\n        Array of templates that will appear in the context menu, Each\n        template is a json object precreated that can be added as a\n        object value to any node in your document.')
    value = param.Parameter(default={}, doc='\n        JSON data to be edited.')
    _rename: ClassVar[Mapping[str, str | None]] = {'name': None, 'value': 'data'}

    def _get_model(self, doc, root=None, parent=None, comm=None):
        if self._widget_type is None:
            self._widget_type = lazy_load('panel.models.jsoneditor', 'JSONEditor', isinstance(comm, JupyterComm))
        model = super()._get_model(doc, root, parent, comm)
        return model
from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Nullable, Required, String
from ..model import Model
class ImportedStyleSheet(StyleSheet):
    """ Imported stylesheet equivalent to ``<link rel="stylesheet" href="${url}">``.

    .. note::
        Depending on the context, this stylesheet will be appended either to
        the the parent shadow root, if used in a component, or otherwise to
        the ``<head>`` element. If you want to append globally regardless of
        the context, use ``GlobalImportedStyleSheet`` instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    url = Required(String, help='\n    The location of an external stylesheet.\n    ')
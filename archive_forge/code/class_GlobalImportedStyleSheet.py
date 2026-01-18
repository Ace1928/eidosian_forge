from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Nullable, Required, String
from ..model import Model
class GlobalImportedStyleSheet(ImportedStyleSheet):
    """ An imported stylesheet that's appended to the ``<head>`` element.

    .. note::
        A stylesheet will be appended only once, regardless of how
        many times it's being used in other models.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
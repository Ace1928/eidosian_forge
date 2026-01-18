from __future__ import annotations
import logging # isort:skip
import pathlib
from typing import TYPE_CHECKING, Any as any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import INVALID_PROPERTY_VALUE, NOT_A_PROPERTY_OF
from ..model import Model
class OpenDialog(Callback):
    """ Open a dialog box. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dialog = Required(Instance('.models.ui.Dialog'), help="\n    A dialog instance to open.\n\n    This will either build and display an new dialog view or re-open an\n    existing dialog view if it was hidden.\n\n    .. note::\n        To display multiple instances of dialog, one needs to clone the\n        dialog's model and use another instance of ``OpenDialog``.\n    ")
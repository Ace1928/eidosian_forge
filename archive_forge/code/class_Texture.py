from __future__ import annotations
import logging # isort:skip
from ..core.enums import TextureRepetition
from ..core.has_props import abstract
from ..core.properties import Enum, Required, String
from ..model import Model
@abstract
class Texture(Model):
    """ Base class for ``Texture`` models that represent fill patterns.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    repetition = Enum(TextureRepetition, default='repeat', help='\n\n    ')
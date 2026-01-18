from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class Ressources:
    """
    Use :class: `Resources` instead.

    .. deprecated:: 5.0.0
    """

    @classmethod
    @property
    def EXT_G_STATE(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/ExtGState'

    @classmethod
    @property
    def COLOR_SPACE(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/ColorSpace'

    @classmethod
    @property
    def PATTERN(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/Pattern'

    @classmethod
    @property
    def SHADING(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/Shading'

    @classmethod
    @property
    def XOBJECT(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/XObject'

    @classmethod
    @property
    def FONT(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/Font'

    @classmethod
    @property
    def PROC_SET(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/ProcSet'

    @classmethod
    @property
    def PROPERTIES(cls) -> str:
        deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
        return '/Properties'
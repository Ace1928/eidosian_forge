from __future__ import annotations
from collections.abc import Sequence
import inspect
from typing import Any, ClassVar, Protocol
from .common.utils import escapeHtml, unescapeAll
from .token import Token
from .utils import EnvType, OptionsDict
class RendererProtocol(Protocol):
    __output__: ClassVar[str]

    def render(self, tokens: Sequence[Token], options: OptionsDict, env: EnvType) -> Any:
        ...
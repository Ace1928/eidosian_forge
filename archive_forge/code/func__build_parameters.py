from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def _build_parameters(server_no_context_takeover: bool, client_no_context_takeover: bool, server_max_window_bits: Optional[int], client_max_window_bits: Optional[Union[int, bool]]) -> List[ExtensionParameter]:
    """
    Build a list of ``(name, value)`` pairs for some compression parameters.

    """
    params: List[ExtensionParameter] = []
    if server_no_context_takeover:
        params.append(('server_no_context_takeover', None))
    if client_no_context_takeover:
        params.append(('client_no_context_takeover', None))
    if server_max_window_bits:
        params.append(('server_max_window_bits', str(server_max_window_bits)))
    if client_max_window_bits is True:
        params.append(('client_max_window_bits', None))
    elif client_max_window_bits:
        params.append(('client_max_window_bits', str(client_max_window_bits)))
    return params
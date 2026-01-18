from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def _extract_parameters(params: Sequence[ExtensionParameter], *, is_server: bool) -> Tuple[bool, bool, Optional[int], Optional[Union[int, bool]]]:
    """
    Extract compression parameters from a list of ``(name, value)`` pairs.

    If ``is_server`` is :obj:`True`, ``client_max_window_bits`` may be
    provided without a value. This is only allowed in handshake requests.

    """
    server_no_context_takeover: bool = False
    client_no_context_takeover: bool = False
    server_max_window_bits: Optional[int] = None
    client_max_window_bits: Optional[Union[int, bool]] = None
    for name, value in params:
        if name == 'server_no_context_takeover':
            if server_no_context_takeover:
                raise exceptions.DuplicateParameter(name)
            if value is None:
                server_no_context_takeover = True
            else:
                raise exceptions.InvalidParameterValue(name, value)
        elif name == 'client_no_context_takeover':
            if client_no_context_takeover:
                raise exceptions.DuplicateParameter(name)
            if value is None:
                client_no_context_takeover = True
            else:
                raise exceptions.InvalidParameterValue(name, value)
        elif name == 'server_max_window_bits':
            if server_max_window_bits is not None:
                raise exceptions.DuplicateParameter(name)
            if value in _MAX_WINDOW_BITS_VALUES:
                server_max_window_bits = int(value)
            else:
                raise exceptions.InvalidParameterValue(name, value)
        elif name == 'client_max_window_bits':
            if client_max_window_bits is not None:
                raise exceptions.DuplicateParameter(name)
            if is_server and value is None:
                client_max_window_bits = True
            elif value in _MAX_WINDOW_BITS_VALUES:
                client_max_window_bits = int(value)
            else:
                raise exceptions.InvalidParameterValue(name, value)
        else:
            raise exceptions.InvalidParameterName(name)
    return (server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits)
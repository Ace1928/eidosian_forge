from __future__ import annotations
import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
from langchain_core.outputs import (
def _create_message_dicts(self, messages: List[BaseMessage]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    params: Dict[str, Any] = {}
    message_dicts = [self._convert_message_to_dict(m) for m in messages]
    return (message_dicts, params)
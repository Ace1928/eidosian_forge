import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class ToolMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model."""
    tool_call_id: str
    'Tool call that this message is responding to.'
    type: Literal['tool'] = 'tool'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']
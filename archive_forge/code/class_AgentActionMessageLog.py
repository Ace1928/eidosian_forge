from __future__ import annotations
import json
from typing import Any, List, Literal, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class AgentActionMessageLog(AgentAction):
    message_log: Sequence[BaseMessage]
    'Similar to log, this can be used to pass along extra\n    information about what exact messages were predicted by the LLM\n    before parsing out the (tool, tool_input). This is again useful\n    if (tool, tool_input) cannot be used to fully recreate the LLM\n    prediction, and you need that LLM prediction (for future agent iteration).\n    Compared to `log`, this is useful when the underlying LLM is a\n    ChatModel (and therefore returns messages rather than a string).'
    type: Literal['AgentActionMessageLog'] = 'AgentActionMessageLog'
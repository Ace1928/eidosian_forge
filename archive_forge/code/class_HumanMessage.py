from typing import List, Literal
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
class HumanMessage(BaseMessage):
    """Message from a human."""
    example: bool = False
    'Whether this Message is being passed in to the model as part of an example \n        conversation.\n    '
    type: Literal['human'] = 'human'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']
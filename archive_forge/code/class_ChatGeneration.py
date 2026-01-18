from __future__ import annotations
from typing import Any, Dict, List, Literal
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts
class ChatGeneration(Generation):
    """A single chat generation output."""
    text: str = ''
    '*SHOULD NOT BE SET DIRECTLY* The text contents of the output message.'
    message: BaseMessage
    'The message output by the chat model.'
    type: Literal['ChatGeneration'] = 'ChatGeneration'
    'Type is used exclusively for serialization purposes.'

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set the text attribute to be the contents of the message."""
        try:
            text = ''
            if isinstance(values['message'].content, str):
                text = values['message'].content
            elif isinstance(values['message'].content, list):
                for block in values['message'].content:
                    if isinstance(block, str):
                        text = block
                        break
                    elif isinstance(block, dict) and 'text' in block:
                        text = block['text']
                        break
                    else:
                        pass
            else:
                pass
            values['text'] = text
        except (KeyError, AttributeError) as e:
            raise ValueError('Error while initializing ChatGeneration') from e
        return values

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'output']
import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _serialize_conversation(conversation: Dict[str, str]) -> str:
    conversation_as_list = []
    for speaker, message in conversation.items():
        conversation_as_list.append(f'{speaker}: {message}')
    return '\n\n'.join(conversation_as_list)
from functools import partial
import torch
import transformers
from ochat.config.model_config import ModelConfig
from ochat.config.conversation_template import Message, Conversation, ConversationTemplate
import ochat.models
def _v3_2_role_prefix(from_role, condition):
    return f'{condition} {_V3_2_PREFIXES[from_role]}'.strip()
import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file
def _encode_header(message, tokenizer):
    tokens = []
    tokens.extend(tokenizer.encode('<|start_header_id|>'))
    tokens.extend(tokenizer.encode(message['role']))
    tokens.extend(tokenizer.encode('<|end_header_id|>'))
    tokens.extend(tokenizer.encode('\n\n'))
    return tokens
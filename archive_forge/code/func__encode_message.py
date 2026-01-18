import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file
def _encode_message(message, tokenizer):
    tokens = _encode_header(message, tokenizer)
    tokens.extend(tokenizer.encode(message['content'].strip()))
    tokens.extend(tokenizer.encode('<|eot_id|>'))
    return tokens
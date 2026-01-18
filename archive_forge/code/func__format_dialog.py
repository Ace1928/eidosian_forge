import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file
def _format_dialog(dialog, tokenizer):
    tokens = []
    tokens.extend(tokenizer.encode('<|begin_of_text|>'))
    for msg in dialog:
        tokens.extend(_encode_message(msg, tokenizer))
    tokens.extend(_encode_header({'role': 'assistant', 'content': ''}, tokenizer))
    return tokens
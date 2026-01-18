import os
from logging import getLogger
from pathlib import Path
from typing import (
import tiktoken
from tiktoken.load import load_tiktoken_bpe
class ChatFormat:

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens['<|start_header_id|>'])
        tokens.extend(self.tokenizer.encode(message['role'], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens['<|end_header_id|>'])
        tokens.extend(self.tokenizer.encode('\n\n', bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message['content'].strip(), bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens['<|eot_id|>'])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens['<|begin_of_text|>'])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        tokens.extend(self.encode_header({'role': 'assistant', 'content': ''}))
        return tokens
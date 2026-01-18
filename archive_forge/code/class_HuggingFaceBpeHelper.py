from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import final
from parlai.core.build_data import download, make_dir
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
class HuggingFaceBpeHelper(BPEHelper):
    """
    HuggingFace's ByteLevelBPE Tokenizer.

    Fast because Rust.
    """

    def __init__(self, opt: Opt, shared: TShared=None):
        super().__init__(opt, shared)
        self.special_tok_map = {}
        self.add_prefix_space = opt.get('bpe_add_prefix_space', True)
        self.skip_special_tokens = opt.get('hf_skip_special_tokens', True)
        if self.add_prefix_space is None:
            self.add_prefix_space = True
        if opt.get('dict_loaded'):
            dfname = opt['dict_file']
            if os.path.isfile(f'{dfname}-merges.txt'):
                opt['bpe_merge'] = f'{dfname}-merges.txt'
            if os.path.isfile(f'{dfname}-vocab.json'):
                opt['bpe_vocab'] = f'{dfname}-vocab.json'
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError('Please install HuggingFace tokenizer with: pip install tokenizers')
        if self.lower:
            warn_once('Are you sure you want to lower case your BPE dictionary?')
        if self.maxtokens > 0 or self.minfreq > 0:
            raise ValueError('You should not filter vocabulary with using --dict-tokenizer bytelevelbpe (no --dict-minfreq or --dict-maxtokens).')
        if 'bpe_vocab' not in opt:
            raise ValueError('--bpe-vocab is required for loading pretrained tokenizer')
        if 'bpe_merge' not in opt:
            raise ValueError('--bpe-merge is required for loading pretrained tokenizer')
        self.vocab_path = opt['bpe_vocab']
        self.merge_path = opt['bpe_merge']
        if not self.vocab_path or not self.merge_path:
            raise IOError('--bpe-vocab and --bpe-merge are mandatory with --dict-tokenizer bytelevelbpe')
        if not os.path.isfile(self.vocab_path):
            raise IOError(f'File {self.vocab_path} does not exist. --bpe-vocab must be pretrained.')
        if not os.path.isfile(self.merge_path):
            raise IOError(f'File {self.merge_path} does not exist. --bpe-merge must be pretrained.')
        self.tokenizer = ByteLevelBPETokenizer(self.vocab_path, self.merge_path, self.add_prefix_space)

    def helper_encode(self, text: str) -> List[str]:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        return self.tokenizer.encode(text).tokens

    def helper_decode(self, tokens: List[str], token_ids: List[int], delimiter: str) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=self.skip_special_tokens)
        return text

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        """
        Add special tokens to the tokenizer and dict_agent.
        """
        logging.debug(f'adding the following special tokens: {special_tokens}')
        self.tokenizer.add_special_tokens(special_tokens)
        for tok in special_tokens:
            parlai_key = dict_agent[tok]
            hf_key = self.tokenizer.token_to_id(tok)
            self.special_tok_map[parlai_key] = hf_key

    def sync_with_dict(self, dict_agent):
        """
        Sync the dictionary agent with Hugging Face tokenizer's BPE dict.

        Called only once on initialization.
        """
        special_tokens = [dict_agent.null_token, dict_agent.start_token, dict_agent.end_token, dict_agent.unk_token]
        self.add_special_tokens(dict_agent, special_tokens)
        for i in range(self.tokenizer.get_vocab_size() - len(special_tokens)):
            token = self.tokenizer.id_to_token(i)
            dict_agent.add_token(token)
            dict_agent.freq[token] = 1

    def save(self, dir_name: str, file_name: str):
        """
        Save appropriate files.

        :param dir_name:
            directory to save.
        :param file_name:
            file to save.
        """
        self.tokenizer.save(dir_name, file_name)
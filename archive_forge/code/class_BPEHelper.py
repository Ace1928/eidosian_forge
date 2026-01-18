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
class BPEHelper(ABC):
    """
    Abstract BPE Helper.

    BPE Helper subclasses must implement appropriate abstractmethods.
    """

    def __init__(self, opt: Opt, shared: TShared=None):
        """
        Subclasses _should_ override __init__ to initialize other things.
        """
        from parlai.core.dict import DictionaryAgent
        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.maxtokens = opt.get('dict_maxtokens', DictionaryAgent.default_maxtokens)
        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)
        self.opt = opt
        self.debug = opt.get('bpe_debug', False)
        self.add_prefix_space = opt.get('bpe_add_prefix_space', False)

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BPEHelper Arguments')
        parser.add_argument('--bpe-vocab', type=str, help='path to pre-trained tokenizer vocab')
        parser.add_argument('--bpe-merge', type=str, help='path to pre-trained tokenizer merge')
        parser.add_argument('--bpe-add-prefix-space', type='bool', hidden=True, help='add prefix space before encoding')
        parser.add_argument('--hf-skip-special-tokens', hidden=True, type='bool', default=True, help='do not decode special tokens with bytelevelbpe')
        return parser

    @final
    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly

        NOTE: DO NOT OVERRIDE

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        if self.add_prefix_space and (not isinstance(self, HuggingFaceBpeHelper)):
            text = f' {text}'
        return self.helper_encode(text)

    @abstractmethod
    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Subclasses should override this method for encoding.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """

    @final
    def decode(self, tokens: List[str], token_ids: List[int], delimiter: str) -> str:
        """
        Decode list of tokens into a text string.

        NOTE: DO NOT OVERRIDE

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = delimiter.join(tokens)
        if not self.debug:
            text = self.helper_decode(tokens, token_ids, delimiter)
            if self.add_prefix_space:
                assert text.startswith(' ')
                text = text.lstrip(' ')
        return text

    @abstractmethod
    def helper_decode(self, tokens: List[str], token_ids: List[int], delimiter: str) -> str:
        """
        Decode list of tokens into text string.

        Subclasses should override this method for decoding.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """

    @abstractmethod
    def sync_with_dict(self, dict_agent):
        """
        Sync BPE Helper dictionary with dict_agent dict.

        :param dict_agent:
            agent with which we are syncing the dictionary
        """

    def finalize(self, frequencies: Dict[str, int], num_symbols: int, minfreq: int) -> bool:
        """
        Build the codecs.

        Default helpers are pre-trained and thus do not build their own codecs

        :param frequencies:
            dictionary of (token: frequency) pairs
        :param num_symbols:
            Number of BPE symbols. Recommend 30000-40000.  If <= 0, default
            30000 will be used.
        :param minfreq:
            Minimum frequency of a token before forced BPE decomposition. If <=
            0 will use subword-nmt default of 2.

        :return did_finalize:
            return whether codecs are finalized this call.
        """
        return False

    def copy_codecs_file(self, target_file: str):
        """
        Copy the codecs file to a new location.

        Default behavior is to do nothing.

        :param target_file:
            where to copy the codecs.
        """
        pass

    def should_sort(self) -> bool:
        """
        Return whether tokens should be sorted for this particular helper.

        DictionaryAgent sorts tokens upon saving; we don't generally want to sort with
        our pre-trained dictionaries, so default is False.
        """
        return False
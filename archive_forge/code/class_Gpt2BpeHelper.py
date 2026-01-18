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
class Gpt2BpeHelper(BPEHelper):
    """
    BPE Helper for GPT2 Models.

    Original source:
        https://github.com/openai/gpt-2/blob/master/src/encoder.py

    Original license: MIT

    This is a modified implementation from that of fairseq:
        https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/gpt2_bpe_utils.py

    Fairseq license: MIT
    """
    DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    ERRORS_METHOD = 'replace'

    def __init__(self, opt: Opt, shared: TShared=None):
        """
        Override init to build the data.
        """
        super().__init__(opt, shared)
        if self.lower:
            warn_once('Are you sure you want to lower case your BPE dictionary?')
        if self.maxtokens > 0 or self.minfreq > 0:
            raise ValueError('You should not filter vocabulary with using --dict-tokenizer bytelevelbpe (no --dict-minfreq or --dict-maxtokens).')
        bpe_data, json_path = self._build_data()
        self.encoder: Dict[str, str] = self._build_encoder(json_path)
        self.decoder: Dict[str, str] = {v: k for k, v in self.encoder.items()}
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        try:
            import regex as re
            self.re = re
        except ImportError:
            raise ImportError('Please install regex with: pip install regex')
        self.pat = self.re.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

    def _build_data(self) -> Tuple[str, str]:
        """
        Build data.

        Maybe download the appropriate data.

        :return (bpe_data, json_path):
            bpe_data and path to encoder json
        """
        data_path = os.path.join(self.opt['datapath'], 'gpt2')
        vocab_path = os.path.join(data_path, 'vocab.bpe')
        json_path = os.path.join(data_path, 'encoder.json')
        if not os.path.isfile(vocab_path) or not os.path.isfile(json_path):
            make_dir(data_path)
            download(self.DEFAULT_VOCAB_BPE, data_path, 'vocab.bpe')
            download(self.DEFAULT_ENCODER_JSON, data_path, 'encoder.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            bpe_data = f.read()
        return (bpe_data, json_path)

    def _build_encoder(self, json_path: str) -> Dict[str, str]:
        """
        Build and return the encoder.

        :param json_path:
            path to encoder json file

        :return:
            encoder, mapping tokens to unicode reps
        """
        with open(json_path, 'r', encoding='utf8') as f:
            encoder = json.load(f)
        for each_token in encoder.keys():
            new_token = ''.join(('\\' + hex(b).lstrip('0') if b > 127 or b < 32 else chr(b) for b in each_token.encode('utf-8')))
            encoder[each_token] = new_token
        return encoder

    @lru_cache()
    def bytes_to_unicode(self) -> Dict[int, str]:
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.

        The reversible bpe codes work on unicode strings. This means you need a large #
        of unicode characters in your vocab if you want to avoid UNKs. When you're at
        something like a 10B token dataset you end up needing around 5K for decent
        coverage. This is a signficant percentage of your normal, say, 32K bpe vocab. To
        avoid that, we want lookup tables between utf-8 bytes and unicode strings. And
        avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs: List[int] = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        cs: List[int] = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        str_cs: List[str] = [chr(n) for n in cs]
        return dict(zip(bs, str_cs))

    def get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """
        Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).

        :param word:
            word to symbolize

        :return pairs:
            set of tuples of symbols
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @lru_cache(maxsize=10240)
    def bpe(self, token: str) -> str:
        """
        Convert token to BPE.

        :param token:
            token to convert

        :return bpe_encoding:
            string bpe encoding
        """
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and (word[i + 1] == second):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        return ' '.join(word)

    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        bpe_tokens: List[str] = []
        for token in self.re.findall(self.pat, text):
            token = ''.join((self.byte_encoder[b] for b in token.encode('utf-8')))
            bpe_tokens.extend((self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')))
        return bpe_tokens

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
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.ERRORS_METHOD)
        return text

    def sync_with_dict(self, dict_agent):
        """
        Sync with dictionary agent.

        Just add all of the tokens to the dict

        NOTE: How does this handle special tokens?

        :param dict_agent:
            A DictionaryAgent instantiation
        """
        for each_token in self.encoder.values():
            dict_agent.add_token(each_token)
            dict_agent.freq[each_token] = 1
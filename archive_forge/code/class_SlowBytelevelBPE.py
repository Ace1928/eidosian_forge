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
class SlowBytelevelBPE(Gpt2BpeHelper):
    """
    Stand-in for HuggingFace if we do not have access to tokenizers.

    Only EVER used for a model used in interactive mode that was previously trained with
    HF BPE.
    """

    def _build_data(self) -> Tuple[str, str]:
        """
        Override to load dicts if they exist.

        :return (bpe_data, json_path):
            bpe_data and path to encoder json
        """
        bpe_data = None
        json_path = ''
        vocab_path = ''
        if self.opt.get('dict_loaded'):
            dfname = self.opt['dict_file']
            if os.path.isfile(f'{dfname}-merges.txt'):
                vocab_path = f'{dfname}-merges.txt'
            if os.path.isfile(f'{dfname}-vocab.json'):
                json_path = f'{dfname}-vocab.json'
        if os.path.isfile(vocab_path) and os.path.isfile(json_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                bpe_data = f.read()
        else:
            bpe_data, json_path = super()._build_data()
        return (bpe_data, json_path)

    def sync_with_dict(self, dict_agent):
        """
        Basically a combination of syncing HF dict with the GPT2 standard.

        It's kinda reversed.

        :param dict_agent:
            Dictionary Agent
        """
        special_tokens = [dict_agent.null_token, dict_agent.start_token, dict_agent.end_token, dict_agent.unk_token]
        dict_agent.tok2ind = {tok: i for tok, i in zip(special_tokens, range(len(special_tokens)))}
        dict_agent.ind2tok = {v: k for k, v in dict_agent.tok2ind.items()}
        for each_token in self.encoder.values():
            dict_agent.add_token(each_token)
            dict_agent.freq[each_token] = 1
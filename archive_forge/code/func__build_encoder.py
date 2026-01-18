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
import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
@dataclass
class Wav2Vec2PhonemeCTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2PhonemeCTCTokenizer`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each charater. Total logit score of the beam associated with
            produced text.
    """
    text: Union[List[str], str]
    char_offsets: Union[List[ListOfDict], ListOfDict] = None
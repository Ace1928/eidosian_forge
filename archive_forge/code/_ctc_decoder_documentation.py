from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset

        Map raw token IDs into corresponding tokens

        Args:
            idxs (LongTensor): raw token IDs generated from decoder

        Returns:
            List: tokens corresponding to the input IDs
        
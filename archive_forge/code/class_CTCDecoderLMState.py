from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
class CTCDecoderLMState(_LMState):
    """Language model state."""

    @property
    def children(self) -> Dict[int, CTCDecoderLMState]:
        """Map of indices to LM states"""
        return super().children

    def child(self, usr_index: int) -> CTCDecoderLMState:
        """Returns child corresponding to usr_index, or creates and returns a new state if input index
        is not found.

        Args:
            usr_index (int): index corresponding to child state

        Returns:
            CTCDecoderLMState: child state corresponding to usr_index
        """
        return super().child(usr_index)

    def compare(self, state: CTCDecoderLMState) -> CTCDecoderLMState:
        """Compare two language model states.

        Args:
            state (CTCDecoderLMState): LM state to compare against

        Returns:
            int: 0 if the states are the same, -1 if self is less, +1 if self is greater.
        """
        pass
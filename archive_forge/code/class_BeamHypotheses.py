from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from .beam_constraints import Constraint, ConstraintListState
class BeamHypotheses:

    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int]=None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0
        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError('When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the BeamScorer class instance at initialization time.')

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor]=None, generated_len: Optional[int]=None):
        """
        Add a new hypothesis to the list.
        """
        if generated_len is not None:
            score = sum_logprobs / generated_len ** self.length_penalty
        else:
            score = sum_logprobs / hyp.shape[-1] ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int]=0) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        if self.early_stopping is True:
            return True
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        else:
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError('max_length is not larger than decoder prompt length')
                highest_attainable_score = best_sum_logprobs / (self.max_length - decoder_prompt_len) ** self.length_penalty
            else:
                highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
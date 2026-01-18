from collections import defaultdict
from typing import DefaultDict, List, Optional, Type, Union
import torch
from pydantic import BaseModel
from transformers import Pipeline, PreTrainedTokenizerBase
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str
class RegexPrefixAllowedTokens:
    """Bias transformers generation based on a regular expression.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, regex_string: str, tokenizer_or_pipe: Union[PreTrainedTokenizerBase, Pipeline]):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression.
        tokenizer_or_pipe
            The tokenizer of the model, or the pipeline object.

        Raises
        ------
        ValueError
            If the `tokenizer_or_pipe` parameter is not a tokenizer or a pipeline.
        """
        if isinstance(tokenizer_or_pipe, Pipeline):
            tokenizer = tokenizer_or_pipe.tokenizer
        elif isinstance(tokenizer_or_pipe, PreTrainedTokenizerBase):
            tokenizer = tokenizer_or_pipe
        else:
            raise ValueError('The tokenizer_or_pipe parameter must be a tokenizer or a pipeline.')
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        self.fsm = RegexGuide(regex_string=regex_string, tokenizer=tokenizer)
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)
        self._prefix = [-1]

    def __call__(self, batch_id: int, sent: torch.Tensor) -> List[int]:
        """Use the FSM to bias the logits before sampling the next token.

        Parameters
        ----------
        batch_id
            The index of the current batch.
        sent
            The tokens of the current sentence.

        Returns
        -------
        List[int]
            The indices of the tokens that are allowed to be sampled next.
        """
        input_ids = sent.tolist()
        if input_ids[:len(self._prefix)] != self._prefix:
            self._fsm_state = defaultdict(int)
            self._prefix = input_ids
            seq_id = hash(tuple([]))
        else:
            input_ids = input_ids[len(self._prefix):]
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            seq_id = hash(tuple(input_ids))
            self._fsm_state[seq_id] = self.fsm.get_next_state(state=self._fsm_state[last_seq_id], token_id=last_token)
        allowed_tokens = self.fsm.get_next_instruction(state=self._fsm_state[seq_id]).tokens
        return allowed_tokens
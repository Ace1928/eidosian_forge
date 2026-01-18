import warnings
from typing import TYPE_CHECKING, List, NewType
from outlines.fsm.guide import CFGGuide, RegexGuide, StopAtEOSGuide
class RegexFSM(RegexGuide):
    """FSM to generate text that is in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer):
        warnings.warn(UserWarning('The `RegexFSM` interface is deprecated and will be removed on 2024-06-01. Please use `RegexGuide` instead.'))
        super().__init__(regex_string, tokenizer)

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        next_instruction = self.get_next_instruction(state)
        return next_instruction.tokens

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        return FSMState(self.get_next_state(state, token_id))
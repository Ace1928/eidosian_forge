import warnings
from typing import TYPE_CHECKING, List, NewType
from outlines.fsm.guide import CFGGuide, RegexGuide, StopAtEOSGuide
class StopAtEosFSM(StopAtEOSGuide):
    """FSM to generate text until EOS has been generated."""

    def __init__(self, tokenizer: 'Tokenizer'):
        warnings.warn(UserWarning('The `StopAtTokenFSM` interface is deprecated and will be removed on 2024-06-01. Please use `StopAtEOSGuide` instead.'))
        super().__init__(tokenizer)

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        next_instruction = self.get_next_instruction(state)
        return next_instruction.tokens

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        return FSMState(self.get_next_state(state, token_id))
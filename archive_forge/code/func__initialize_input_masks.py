from ...rnn import BidirectionalCell, SequentialRNNCell, ModifierCell, HybridRecurrentCell
from ...rnn.rnn_cell import _format_sequence, _get_begin_state, _mask_sequence_variable_length
from ... import tensor_types
from ....base import _as_list
def _initialize_input_masks(self, F, inputs, states):
    if self.drop_states and self.drop_states_mask is None:
        self.drop_states_mask = F.Dropout(F.ones_like(states[0]), p=self.drop_states)
    if self.drop_inputs and self.drop_inputs_mask is None:
        self.drop_inputs_mask = F.Dropout(F.ones_like(inputs), p=self.drop_inputs)
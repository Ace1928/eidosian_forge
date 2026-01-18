import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _maybe_reset_dropout_masks(self, cell):
    if isinstance(cell, DropoutRNNCell):
        cell.reset_dropout_mask()
        cell.reset_recurrent_dropout_mask()
    if isinstance(cell, StackedRNNCells):
        for c in cell.cells:
            self._maybe_reset_dropout_masks(c)
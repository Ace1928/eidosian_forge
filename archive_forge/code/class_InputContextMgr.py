from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
class InputContextMgr:
    """
        A helper class to manage an input context stack.

        The class is designed to be used in a recursion with nested
        'with' statements.

        Parameters
        ----------
        builder : CalciteBuilder
            An outer builder.
        input_frames : list of DFAlgNode
            Input nodes for the new context.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Attributes
        ----------
        builder : CalciteBuilder
            An outer builder.
        input_frames : list of DFAlgNode
            Input nodes for the new context.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.
        """

    def __init__(self, builder, input_frames, input_nodes):
        self.builder = builder
        self.input_frames = input_frames
        self.input_nodes = input_nodes

    def __enter__(self):
        """
            Push new input context into the input context stack.

            Returns
            -------
            InputContext
                New input context.
            """
        self.builder._input_ctx_stack.append(self.builder.InputContext(self.input_frames, self.input_nodes))
        return self.builder._input_ctx_stack[-1]

    def __exit__(self, type, value, traceback):
        """
            Pop current input context.

            Parameters
            ----------
            type : Any
                An exception type.
            value : Any
                An exception value.
            traceback : Any
                A traceback.
            """
        self.builder._input_ctx_stack.pop()
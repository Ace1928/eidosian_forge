import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
class DFAlgNode(abc.ABC):
    """
    A base class for dataframe algebra tree node.

    A dataframe algebra tree is used to describe how dataframe is computed.

    Attributes
    ----------
    input : list of DFAlgNode, optional
        Holds child nodes.
    """

    @abc.abstractmethod
    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        DFAlgNode
        """
        pass

    def walk_dfs(self, cb, *args, **kwargs):
        """
        Perform a depth-first walk over a tree.

        Walk over an input in the depth-first order and call a callback function
        for each node.

        Parameters
        ----------
        cb : callable
            A callback function.
        *args : list
            Arguments for the callback.
        **kwargs : dict
            Keyword arguments for the callback.
        """
        if hasattr(self, 'input'):
            for i in self.input:
                i._op.walk_dfs(cb, *args, **kwargs)
        cb(self, *args, **kwargs)

    def collect_partitions(self):
        """
        Collect all partitions participating in a tree.

        Returns
        -------
        list
            A list of collected partitions.
        """
        partitions = []
        self.walk_dfs(lambda a, b: a._append_partitions(b), partitions)
        return partitions

    def collect_frames(self):
        """
        Collect all frames participating in a tree.

        Returns
        -------
        list
            A list of collected frames.
        """
        frames = []
        self.walk_dfs(lambda a, b: a._append_frames(b), frames)
        return frames

    def require_executed_base(self) -> bool:
        """
        Check if materialization of input frames is required.

        Returns
        -------
        bool
        """
        return False

    def can_execute_hdk(self) -> bool:
        """
        Check for possibility of HDK execution.

        Check if the computation can be executed using an HDK query.

        Returns
        -------
        bool
        """
        return True

    def can_execute_arrow(self) -> bool:
        """
        Check for possibility of Arrow execution.

        Check if the computation can be executed using
        the Arrow API instead of HDK query.

        Returns
        -------
        bool
        """
        return False

    def execute_arrow(self, arrow_input: Union[None, pa.Table, List[pa.Table]]) -> pa.Table:
        """
        Compute the frame data using the Arrow API.

        Parameters
        ----------
        arrow_input : None, pa.Table or list of pa.Table
            The input, converted to arrow.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        raise RuntimeError(f'Arrow execution is not supported by {type(self)}')

    def _append_partitions(self, partitions):
        """
        Append all used by the node partitions to `partitions` list.

        The default implementation is no-op. This method should be
        overriden by all nodes referencing frame's partitions.

        Parameters
        ----------
        partitions : list
            Output list of partitions.
        """
        pass

    def _append_frames(self, frames):
        """
        Append all used by the node frames to `frames` list.

        The default implementation is no-op. This method should be
        overriden by all nodes referencing frames.

        Parameters
        ----------
        frames : list
            Output list of frames.
        """
        pass

    def __repr__(self):
        """
        Return a string representation of the tree.

        Returns
        -------
        str
        """
        return self.dumps()

    def dump(self, prefix=''):
        """
        Dump the tree.

        Parameters
        ----------
        prefix : str, default: ''
            A prefix to add at each string of the dump.
        """
        print(self.dumps(prefix))

    def dumps(self, prefix=''):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str, default: ''
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return self._prints(prefix)

    @abc.abstractmethod
    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        pass

    def _prints_input(self, prefix):
        """
        Return a string representation of node's operands.

        A helper method for `_prints` implementation in derived classes.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        res = ''
        if hasattr(self, 'input'):
            for i, node in enumerate(self.input):
                if isinstance(node._op, FrameNode):
                    res += f'{prefix}input[{i}]: {node._op}\n'
                else:
                    res += f'{prefix}input[{i}]:\n' + node._op._prints(prefix + '  ')
        return res
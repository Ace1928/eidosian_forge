import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
class SplitBase(metaclass=abc.ABCMeta):
    """Abstract base class for Split compositionality.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.

    There are three parts to the composition:
        1) The splits are composed (defined, merged, split,...) together before
             calling the `.as_dataset()` function. This is done with the `__add__`,
             `__getitem__`, which return a tree of `SplitBase` (whose leaf
             are the `NamedSplit` objects)

        ```
        split = datasets.Split.TRAIN + datasets.Split.TEST.subsplit(datasets.percent[:50])
        ```

        2) The `SplitBase` is forwarded to the `.as_dataset()` function
             to be resolved into actual read instruction. This is done by the
             `.get_read_instruction()` method which takes the real dataset splits
             (name, number of shards,...) and parse the tree to return a
             `SplitReadInstruction()` object

        ```
        read_instruction = split.get_read_instruction(self.info.splits)
        ```

        3) The `SplitReadInstruction` is then used in the `tf.data.Dataset` pipeline
             to define which files to read and how to skip examples within file.

    """

    @abc.abstractmethod
    def get_read_instruction(self, split_dict):
        """Parse the descriptor tree and compile all read instructions together.

        Args:
            split_dict: `dict`, The `dict[split_name, SplitInfo]` of the dataset

        Returns:
            split_read_instruction: `SplitReadInstruction`
        """
        raise NotImplementedError('Abstract method')

    def __eq__(self, other):
        """Equality: datasets.Split.TRAIN == 'train'."""
        if isinstance(other, (NamedSplit, str)):
            return False
        raise NotImplementedError('Equality is not implemented between merged/sub splits.')

    def __ne__(self, other):
        """InEquality: datasets.Split.TRAIN != 'test'."""
        return not self.__eq__(other)

    def __add__(self, other):
        """Merging: datasets.Split.TRAIN + datasets.Split.TEST."""
        return _SplitMerged(self, other)

    def subsplit(self, arg=None, k=None, percent=None, weighted=None):
        """Divides this split into subsplits.

        There are 3 ways to define subsplits, which correspond to the 3
        arguments `k` (get `k` even subsplits), `percent` (get a slice of the
        dataset with `datasets.percent`), and `weighted` (get subsplits with proportions
        specified by `weighted`).

        Example::

        ```
        # 50% train, 50% test
        train, test = split.subsplit(k=2)
        # 50% train, 25% test, 25% validation
        train, test, validation = split.subsplit(weighted=[2, 1, 1])
        # Extract last 20%
        subsplit = split.subsplit(datasets.percent[-20:])
        ```

        Warning: k and weighted will be converted into percent which mean that
        values below the percent will be rounded up or down. The final split may be
        bigger to deal with remainders. For instance:

        ```
        train, test, valid = split.subsplit(k=3)  # 33%, 33%, 34%
        s1, s2, s3, s4 = split.subsplit(weighted=[2, 2, 1, 1])  # 33%, 33%, 16%, 18%
        ```

        Args:
            arg: If no kwargs are given, `arg` will be interpreted as one of
                `k`, `percent`, or `weighted` depending on the type.
                For example:
                ```
                split.subsplit(10)  # Equivalent to split.subsplit(k=10)
                split.subsplit(datasets.percent[:-20])  # percent=datasets.percent[:-20]
                split.subsplit([1, 1, 2])  # weighted=[1, 1, 2]
                ```
            k: `int` If set, subdivide the split into `k` equal parts.
            percent: `datasets.percent slice`, return a single subsplit corresponding to
                a slice of the original split. For example:
                `split.subsplit(datasets.percent[-20:])  # Last 20% of the dataset`.
            weighted: `list[int]`, return a list of subsplits whose proportions match
                the normalized sum of the list. For example:
                `split.subsplit(weighted=[1, 1, 2])  # 25%, 25%, 50%`.

        Returns:
            A subsplit or list of subsplits extracted from this split object.
        """
        if sum((bool(x) for x in (arg, k, percent, weighted))) != 1:
            raise ValueError('Only one argument of subsplit should be set.')
        if isinstance(arg, int):
            k = arg
        elif isinstance(arg, slice):
            percent = arg
        elif isinstance(arg, list):
            weighted = arg
        if not (k or percent or weighted):
            raise ValueError(f'Invalid split argument {arg}. Only list, slice and int supported. One of k, weighted or percent should be set to a non empty value.')

        def assert_slices_coverage(slices):
            assert sum((list(range(*s.indices(100))) for s in slices), []) == list(range(100))
        if k:
            if not 0 < k <= 100:
                raise ValueError(f'Subsplit k should be between 0 and 100, got {k}')
            shift = 100 // k
            slices = [slice(i * shift, (i + 1) * shift) for i in range(k)]
            slices[-1] = slice(slices[-1].start, 100)
            assert_slices_coverage(slices)
            return tuple((_SubSplit(self, s) for s in slices))
        elif percent:
            return _SubSplit(self, percent)
        elif weighted:
            total = sum(weighted)
            weighted = [100 * x // total for x in weighted]
            start = 0
            stop = 0
            slices = []
            for v in weighted:
                stop += v
                slices.append(slice(start, stop))
                start = stop
            slices[-1] = slice(slices[-1].start, 100)
            assert_slices_coverage(slices)
            return tuple((_SubSplit(self, s) for s in slices))
        else:
            raise ValueError('Could not determine the split')
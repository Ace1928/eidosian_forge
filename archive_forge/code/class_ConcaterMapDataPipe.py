from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from typing import Sized, Tuple, TypeVar
@functional_datapipe('concat')
class ConcaterMapDataPipe(MapDataPipe):
    """
    Concatenate multiple Map DataPipes (functional name: ``concat``).

    The new index of is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
        datapipes: Map DataPipes being concatenated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """
    datapipes: Tuple[MapDataPipe]

    def __init__(self, *datapipes: MapDataPipe):
        if len(datapipes) == 0:
            raise ValueError('Expected at least one DataPipe, but got nothing')
        if not all((isinstance(dp, MapDataPipe) for dp in datapipes)):
            raise TypeError('Expected all inputs to be `MapDataPipe`')
        if not all((isinstance(dp, Sized) for dp in datapipes)):
            raise TypeError('Expected all inputs to be `Sized`')
        self.datapipes = datapipes

    def __getitem__(self, index) -> T_co:
        offset = 0
        for dp in self.datapipes:
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                offset += len(dp)
        raise IndexError(f'Index {index} is out of range.')

    def __len__(self) -> int:
        return sum((len(dp) for dp in self.datapipes))
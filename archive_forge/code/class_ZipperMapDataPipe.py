from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from typing import Sized, Tuple, TypeVar
@functional_datapipe('zip')
class ZipperMapDataPipe(MapDataPipe[Tuple[T_co, ...]]):
    """
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """
    datapipes: Tuple[MapDataPipe[T_co], ...]

    def __init__(self, *datapipes: MapDataPipe[T_co]) -> None:
        if len(datapipes) == 0:
            raise ValueError('Expected at least one DataPipe, but got nothing')
        if not all((isinstance(dp, MapDataPipe) for dp in datapipes)):
            raise TypeError('Expected all inputs to be `MapDataPipe`')
        if not all((isinstance(dp, Sized) for dp in datapipes)):
            raise TypeError('Expected all inputs to be `Sized`')
        self.datapipes = datapipes

    def __getitem__(self, index) -> Tuple[T_co, ...]:
        res = []
        for dp in self.datapipes:
            try:
                res.append(dp[index])
            except IndexError as e:
                raise IndexError(f'Index {index} is out of range for one of the input MapDataPipes {dp}.') from e
        return tuple(res)

    def __len__(self) -> int:
        return min((len(dp) for dp in self.datapipes))
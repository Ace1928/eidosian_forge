import itertools
def _CheckCombArgument(n: int, k: int) -> None:
    if not isinstance(n, int) or not isinstance(k, int):
        raise ValueError(f'n ({n}) and k ({k}) must be positive integers')
    if n < 0:
        raise ValueError(f'n ({n}) must be a positive integer')
    if k < 0:
        raise ValueError(f'k ({k}) must be a positive integer')
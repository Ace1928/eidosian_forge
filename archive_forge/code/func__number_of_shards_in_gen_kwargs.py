from typing import List
import numpy as np
def _number_of_shards_in_gen_kwargs(gen_kwargs: dict) -> int:
    """Return the number of possible shards according to the input gen_kwargs"""
    lists_lengths = {key: len(value) for key, value in gen_kwargs.items() if isinstance(value, list)}
    if len(set(lists_lengths.values())) > 1:
        raise RuntimeError('Sharding is ambiguous for this dataset: ' + "we found several data sources lists of different lengths, and we don't know over which list we should parallelize:\n" + '\n'.join((f'\t- key {key} has length {length}' for key, length in lists_lengths.items())) + "\nTo fix this, check the 'gen_kwargs' and make sure to use lists only for data sources, " + 'and use tuples otherwise. In the end there should only be one single list, or several lists with the same length.')
    max_length = max(lists_lengths.values(), default=0)
    return max(1, max_length)
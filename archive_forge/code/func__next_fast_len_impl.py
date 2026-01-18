from typing import Dict, List, Tuple
import math
def _next_fast_len_impl(n, primes):
    if len(primes) == 0:
        return math.inf
    result = _next_fast_len_cache.get((n, primes), None)
    if result is None:
        if n == 1:
            result = 1
        else:
            p = primes[0]
            result = min(_next_fast_len_impl((n + p - 1) // p, primes) * p, _next_fast_len_impl(n, primes[1:]))
        _next_fast_len_cache[n, primes] = result
    return result
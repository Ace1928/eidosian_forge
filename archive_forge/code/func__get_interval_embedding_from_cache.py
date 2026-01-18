import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _get_interval_embedding_from_cache(nf, RIF, cache):
    """
    Evaluate RIF(nf.gen_embedding()) where RIF is a RealIntervalField with
    some precision. This is a real interval that is guaranteed to contain the
    preferred root of the defining polynomial of the number field.

    To avoid re-evaluation, use cache which is (a reference) to a python
    dictionary.

    The idea is that while working over one number field, all instances of
    (_Factorized)SqrtLinCombination have a reference to the same (shared) python
    dictionary and fill it in as needed.

    Unfortunately, the reference to the cache needs to passed down along a lot
    of places. There might be a nicer mechanism for doing this.
    """
    if cache is not None and 'gen_embedding' in cache:
        gen_embedding = cache['gen_embedding']
    else:
        gen_embedding = nf.gen_embedding()
        if cache is not None:
            cache['gen_embedding'] = gen_embedding
    prec = RIF.prec()
    if cache is not None and prec in cache:
        return cache[prec]
    interval = RIF(gen_embedding)
    if cache is not None:
        cache[prec] = interval
    return interval
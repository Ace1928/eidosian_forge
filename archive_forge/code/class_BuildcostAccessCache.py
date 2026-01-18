from time import time as gettime
class BuildcostAccessCache(BasicCache):
    """ A BuildTime/Access-counting cache implementation.
        the weight of a value is computed as the product of

            num-accesses-of-a-value * time-to-build-the-value

        The values with the least such weights are evicted
        if the cache maxentries threshold is superceded.
        For implementation flexibility more than one object
        might be evicted at a time.
    """

    def _build(self, key, builder):
        start = gettime()
        val = builder()
        end = gettime()
        return WeightedCountingEntry(val, end - start)
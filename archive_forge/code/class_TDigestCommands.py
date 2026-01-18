from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
class TDigestCommands:

    def create(self, key, compression=100):
        """
        Allocate the memory and initialize the t-digest.
        For more information see `TDIGEST.CREATE <https://redis.io/commands/tdigest.create>`_.
        """
        return self.execute_command(TDIGEST_CREATE, key, 'COMPRESSION', compression)

    def reset(self, key):
        """
        Reset the sketch `key` to zero - empty out the sketch and re-initialize it.
        For more information see `TDIGEST.RESET <https://redis.io/commands/tdigest.reset>`_.
        """
        return self.execute_command(TDIGEST_RESET, key)

    def add(self, key, values):
        """
        Adds one or more observations to a t-digest sketch `key`.

        For more information see `TDIGEST.ADD <https://redis.io/commands/tdigest.add>`_.
        """
        return self.execute_command(TDIGEST_ADD, key, *values)

    def merge(self, destination_key, num_keys, *keys, compression=None, override=False):
        """
        Merges all of the values from `keys` to 'destination-key' sketch.
        It is mandatory to provide the `num_keys` before passing the input keys and
        the other (optional) arguments.
        If `destination_key` already exists its values are merged with the input keys.
        If you wish to override the destination key contents use the `OVERRIDE` parameter.

        For more information see `TDIGEST.MERGE <https://redis.io/commands/tdigest.merge>`_.
        """
        params = [destination_key, num_keys, *keys]
        if compression is not None:
            params.extend(['COMPRESSION', compression])
        if override:
            params.append('OVERRIDE')
        return self.execute_command(TDIGEST_MERGE, *params)

    def min(self, key):
        """
        Return minimum value from the sketch `key`. Will return DBL_MAX if the sketch is empty.
        For more information see `TDIGEST.MIN <https://redis.io/commands/tdigest.min>`_.
        """
        return self.execute_command(TDIGEST_MIN, key)

    def max(self, key):
        """
        Return maximum value from the sketch `key`. Will return DBL_MIN if the sketch is empty.
        For more information see `TDIGEST.MAX <https://redis.io/commands/tdigest.max>`_.
        """
        return self.execute_command(TDIGEST_MAX, key)

    def quantile(self, key, quantile, *quantiles):
        """
        Returns estimates of one or more cutoffs such that a specified fraction of the
        observations added to this t-digest would be less than or equal to each of the
        specified cutoffs. (Multiple quantiles can be returned with one call)
        For more information see `TDIGEST.QUANTILE <https://redis.io/commands/tdigest.quantile>`_.
        """
        return self.execute_command(TDIGEST_QUANTILE, key, quantile, *quantiles)

    def cdf(self, key, value, *values):
        """
        Return double fraction of all points added which are <= value.
        For more information see `TDIGEST.CDF <https://redis.io/commands/tdigest.cdf>`_.
        """
        return self.execute_command(TDIGEST_CDF, key, value, *values)

    def info(self, key):
        """
        Return Compression, Capacity, Merged Nodes, Unmerged Nodes, Merged Weight, Unmerged Weight
        and Total Compressions.
        For more information see `TDIGEST.INFO <https://redis.io/commands/tdigest.info>`_.
        """
        return self.execute_command(TDIGEST_INFO, key)

    def trimmed_mean(self, key, low_cut_quantile, high_cut_quantile):
        """
        Return mean value from the sketch, excluding observation values outside
        the low and high cutoff quantiles.
        For more information see `TDIGEST.TRIMMED_MEAN <https://redis.io/commands/tdigest.trimmed_mean>`_.
        """
        return self.execute_command(TDIGEST_TRIMMED_MEAN, key, low_cut_quantile, high_cut_quantile)

    def rank(self, key, value, *values):
        """
        Retrieve the estimated rank of value (the number of observations in the sketch
        that are smaller than value + half the number of observations that are equal to value).

        For more information see `TDIGEST.RANK <https://redis.io/commands/tdigest.rank>`_.
        """
        return self.execute_command(TDIGEST_RANK, key, value, *values)

    def revrank(self, key, value, *values):
        """
        Retrieve the estimated rank of value (the number of observations in the sketch
        that are larger than value + half the number of observations that are equal to value).

        For more information see `TDIGEST.REVRANK <https://redis.io/commands/tdigest.revrank>`_.
        """
        return self.execute_command(TDIGEST_REVRANK, key, value, *values)

    def byrank(self, key, rank, *ranks):
        """
        Retrieve an estimation of the value with the given rank.

        For more information see `TDIGEST.BY_RANK <https://redis.io/commands/tdigest.by_rank>`_.
        """
        return self.execute_command(TDIGEST_BYRANK, key, rank, *ranks)

    def byrevrank(self, key, rank, *ranks):
        """
        Retrieve an estimation of the value with the given reverse rank.

        For more information see `TDIGEST.BY_REVRANK <https://redis.io/commands/tdigest.by_revrank>`_.
        """
        return self.execute_command(TDIGEST_BYREVRANK, key, rank, *ranks)
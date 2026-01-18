from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
class CMSCommands:
    """Count-Min Sketch Commands"""

    def initbydim(self, key, width, depth):
        """
        Initialize a Count-Min Sketch `key` to dimensions (`width`, `depth`) specified by user.
        For more information see `CMS.INITBYDIM <https://redis.io/commands/cms.initbydim>`_.
        """
        return self.execute_command(CMS_INITBYDIM, key, width, depth)

    def initbyprob(self, key, error, probability):
        """
        Initialize a Count-Min Sketch `key` to characteristics (`error`, `probability`) specified by user.
        For more information see `CMS.INITBYPROB <https://redis.io/commands/cms.initbyprob>`_.
        """
        return self.execute_command(CMS_INITBYPROB, key, error, probability)

    def incrby(self, key, items, increments):
        """
        Add/increase `items` to a Count-Min Sketch `key` by ''increments''.
        Both `items` and `increments` are lists.
        For more information see `CMS.INCRBY <https://redis.io/commands/cms.incrby>`_.

        Example:

        >>> cmsincrby('A', ['foo'], [1])
        """
        params = [key]
        self.append_items_and_increments(params, items, increments)
        return self.execute_command(CMS_INCRBY, *params)

    def query(self, key, *items):
        """
        Return count for an `item` from `key`. Multiple items can be queried with one call.
        For more information see `CMS.QUERY <https://redis.io/commands/cms.query>`_.
        """
        return self.execute_command(CMS_QUERY, key, *items)

    def merge(self, destKey, numKeys, srcKeys, weights=[]):
        """
        Merge `numKeys` of sketches into `destKey`. Sketches specified in `srcKeys`.
        All sketches must have identical width and depth.
        `Weights` can be used to multiply certain sketches. Default weight is 1.
        Both `srcKeys` and `weights` are lists.
        For more information see `CMS.MERGE <https://redis.io/commands/cms.merge>`_.
        """
        params = [destKey, numKeys]
        params += srcKeys
        self.append_weights(params, weights)
        return self.execute_command(CMS_MERGE, *params)

    def info(self, key):
        """
        Return width, depth and total count of the sketch.
        For more information see `CMS.INFO <https://redis.io/commands/cms.info>`_.
        """
        return self.execute_command(CMS_INFO, key)
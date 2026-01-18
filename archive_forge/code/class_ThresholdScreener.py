from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
class ThresholdScreener(SimilarityScreener):
    """ Used to return all compounds that have a similarity
      to the probe beyond a threshold value

     **Notes**:

       - This is as lazy as possible, so the data source isn't
         queried until the client asks for a hit.

       - In addition to being lazy, this class is as thin as possible.
         (Who'd have thought it was possible!)
         Hits are *not* stored locally, so if a client resets
         the iteration and starts over, the same amount of work must
         be done to retrieve the hits.

       - The thinness and laziness forces us to support only forward
         iteration (not random access)

    """

    def __init__(self, threshold, **kwargs):
        SimilarityScreener.__init__(self, **kwargs)
        self.threshold = threshold
        self.dataIter = iter(self.dataSource)

    def _nextMatch(self):
        """ *Internal use only* """
        done = 0
        res = None
        sim = 0
        while not done:
            obj = next(self.dataIter)
            fp = self.fingerprinter(obj)
            sim = DataStructs.FingerprintSimilarity(fp, self.probe, self.metric)
            if sim >= self.threshold:
                res = obj
                done = 1
        return (sim, res)

    def Reset(self):
        """ used to reset our internal state so that iteration
          starts again from the beginning
        """
        self.dataSource.reset()
        self.dataIter = iter(self.dataSource)

    def __iter__(self):
        """ returns an iterator for this screener
        """
        self.Reset()
        return self

    def next(self):
        """ required part of iterator interface """
        return self._nextMatch()
    __next__ = next
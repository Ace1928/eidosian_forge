from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
class SimilarityScreener(object):
    """  base class

     important attributes:
        probe: the probe fingerprint against which we screen.

        metric: a function that takes two arguments and returns a similarity
                measure between them

        dataSource: the source pool from which to draw, needs to support
                a next() method

        fingerprinter: a function that takes a molecule and returns a
               fingerprint of the appropriate format


      **Notes**
         subclasses must support either an iterator interface
         or __len__ and __getitem__
    """

    def __init__(self, probe=None, metric=None, dataSource=None, fingerprinter=None):
        self.metric = metric
        self.dataSource = dataSource
        self.fingerprinter = fingerprinter
        self.probe = probe

    def Reset(self):
        """ used to reset screeners that behave as iterators """
        pass

    def SetProbe(self, probeFingerprint):
        """ sets our probe fingerprint """
        self.probe = probeFingerprint

    def GetSingleFingerprint(self, probe):
        """ returns a fingerprint for a single probe object

         This is potentially useful in initializing our internal
         probe object.

        """
        return self.fingerprinter(probe)
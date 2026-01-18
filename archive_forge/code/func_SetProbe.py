from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
def SetProbe(self, probeFingerprint):
    """ sets our probe fingerprint """
    self.probe = probeFingerprint
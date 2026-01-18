import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
class _UniqueRandomBitsGenerator(object):

    def __init__(self, nCenters, maxIsomers, rand):
        self.nCenters = nCenters
        self.maxIsomers = maxIsomers
        self.rand = rand
        self.already_seen = set()

    def __iter__(self):
        while len(self.already_seen) < 2 ** self.nCenters:
            bits = self.rand.getrandbits(self.nCenters)
            if bits in self.already_seen:
                continue
            self.already_seen.add(bits)
            yield bits
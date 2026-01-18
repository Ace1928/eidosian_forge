import copy
import struct
from rdkit import DataStructs
def AddVect(self, idx, vect):
    self.__vects[idx] = vect
    self.__needReset = True
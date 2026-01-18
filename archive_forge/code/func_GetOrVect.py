import copy
import struct
from rdkit import DataStructs
def GetOrVect(self):
    if self.__needReset:
        self.Reset()
    return self.__orVect
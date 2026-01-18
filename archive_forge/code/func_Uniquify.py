import copy
import struct
from rdkit import DataStructs
def Uniquify(self, verbose=False):
    obls = {}
    for k, v in self.__vects.items():
        obls[k] = list(v.GetOnBits())
    keys = list(self.__vects.keys())
    nKeys = len(keys)
    keep = list(self.__vects.keys())
    for i in range(nKeys):
        k1 = keys[i]
        if k1 in keep:
            obl1 = obls[k1]
            idx = keys.index(k1)
            for j in range(idx + 1, nKeys):
                k2 = keys[j]
                if k2 in keep:
                    obl2 = obls[k2]
                    if obl1 == obl2:
                        keep.remove(k2)
    self.__needsReset = True
    tmp = {}
    for k in keep:
        tmp[k] = self.__vects[k]
    if verbose:
        print('uniquify:', len(self.__vects), '->', len(tmp))
    self.__vects = tmp
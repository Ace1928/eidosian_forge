from rdkit import Chem
class FragmentMatcher(object):

    def __init__(self):
        self._onPatt = None
        self._offPatts = []

    def AddExclusion(self, sma):
        self._offPatts.append(Chem.MolFromSmarts(sma))

    def Init(self, sma):
        self._onPatt = Chem.MolFromSmarts(sma)

    def GetSMARTS(self):
        pass

    def GetExclusionSMARTS(self):
        pass

    def HasMatch(self, mol):
        if self._onPatt is None:
            return 0
        t = mol.HasSubstructMatch(self._onPatt)
        if not t:
            return 0
        else:
            for patt in self._offPatts:
                if mol.HasSubstructMatch(patt):
                    return 0
        return 1

    def GetMatch(self, mol):
        if self._onPatt is None:
            return None
        return mol.GetSubstructMatch(self._onPatt)

    def GetMatches(self, mol, uniquify=1):
        if self._onPatt is None:
            return None
        return mol.GetSubstructMatches(self._onPatt, uniquify=uniquify)

    def GetBond(self, idx):
        if self._onPatt is None:
            return None
        return self._onPatt.GetBondWithIdx(idx)
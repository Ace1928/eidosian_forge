from rdkit import RDLogger as logging
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.ChemUtils.AlignDepict import AlignDepict
def ConstructSidechains(suppl, sma=None, replace=True, useAll=False):
    if sma:
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            logger.error('could not construct pattern from smarts: %s' % sma, exc_info=True)
            return None
    else:
        patt = None
    if replace:
        replacement = Chem.MolFromSmiles('[*]')
    res = []
    for idx, mol in enumerate(suppl):
        if not mol:
            continue
        if patt:
            if not mol.HasSubstructMatch(patt):
                logger.warning('The substructure pattern did not match sidechain %d. This may result in errors.' % (idx + 1))
            if replace:
                tmp = list(Chem.ReplaceSubstructs(mol, patt, replacement))
                if not useAll:
                    tmp = [tmp[0]]
                for i, entry in enumerate(tmp):
                    entry = MoveDummyNeighborsToBeginning(entry)
                    if not entry:
                        continue
                    entry = entry[0]
                    for propN in mol.GetPropNames():
                        entry.SetProp(propN, mol.GetProp(propN))
                    tmp[i] = (idx + 1, entry)
            else:
                matches = mol.GetSubstructMatches(patt)
                if matches:
                    tmp = [0] * len(matches)
                    for i, match in enumerate(matches):
                        smi = Chem.MolToSmiles(mol, rootedAtAtom=match[0])
                        entry = Chem.MolFromSmiles(smi)
                        for propN in mol.GetPropNames():
                            entry.SetProp(propN, mol.GetProp(propN))
                        tmp[i] = (idx + 1, entry)
                else:
                    tmp = None
        else:
            tmp = [(idx + 1, mol)]
        if tmp:
            res.extend(tmp)
    return res
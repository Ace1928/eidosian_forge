import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import DbFpSupplier, FingerprintMols
from rdkit.DataStructs.TopNContainer import TopNContainer
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def ScreenFingerprints(details, data, mol=None, probeFp=None):
    """ Returns a list of results

  """
    if probeFp is None:
        try:
            probeFp = FingerprintMols.FingerprintMol(mol, **details.__dict__)
        except Exception:
            import traceback
            FingerprintMols.error('Error: problems fingerprinting molecule.\n')
            traceback.print_exc()
            return []
    if not probeFp:
        return []
    if not details.doThreshold and details.topN > 0:
        topN = TopNContainer(details.topN)
    else:
        topN = []
    res = []
    count = 0
    for pt in data:
        fp1 = probeFp
        if not details.noPickle:
            if isinstance(pt, (tuple, list)):
                ID, fp = pt
            else:
                fp = pt
                ID = pt._fieldsFromDb[0]
            score = DataStructs.FingerprintSimilarity(fp1, fp, details.metric)
        else:
            ID, pkl = pt
            score = details.metric(fp1, str(pkl))
        if topN:
            topN.Insert(score, ID)
        elif not details.doThreshold or (details.doThreshold and score >= details.screenThresh):
            res.append((ID, score))
        count += 1
        if hasattr(details, 'stopAfter') and count >= details.stopAfter:
            break
    for score, ID in topN:
        res.append((ID, score))
    return res
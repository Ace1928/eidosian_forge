import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import DbFpSupplier, FingerprintMols
from rdkit.DataStructs.TopNContainer import TopNContainer
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def ScreenInDb(details, mol):
    try:
        probeFp = FingerprintMols.FingerprintMol(mol, **details.__dict__)
    except Exception:
        import traceback
        FingerprintMols.error('Error: problems fingerprinting molecule.\n')
        traceback.print_exc()
        return []
    if details.metric not in (DataStructs.TanimotoSimilarity, DataStructs.DiceSimilarity, DataStructs.CosineSimilarity):
        return ScreenFingerprints(details, data=GetFingerprints(details), mol=mol)
    conn: DbConnect = _ConnectToDatabase(details)
    if details.metric == DataStructs.TanimotoSimilarity:
        func = 'rd_tanimoto'
    elif details.metric == DataStructs.DiceSimilarity:
        func = 'rd_dice'
    elif details.metric == DataStructs.CosineSimilarity:
        func = 'rd_cosine'
    pkl = probeFp.ToBitString()
    extraFields = f'{func}({DbModule.placeHolder},{details.fpColName}) as tani'
    cmd = _ConstructSQL(details, extraFields=extraFields)
    if details.doThreshold:
        cmd = f'select * from ({cmd}) tmp where tani>{details.screenThresh}'
    cmd += ' order by tani desc'
    if not details.doThreshold and details.topN > 0:
        cmd += f' limit {details.topN}'
    curs = conn.GetCursor()
    curs.execute(cmd, (pkl,))
    return curs.fetchall()
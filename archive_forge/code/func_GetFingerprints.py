import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import DbFpSupplier, FingerprintMols
from rdkit.DataStructs.TopNContainer import TopNContainer
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def GetFingerprints(details):
    """ returns an iterable sequence of fingerprints
  each fingerprint will have a _fieldsFromDb member whose first entry is
  the id.

  """
    if details.dbName and details.tableName:
        conn: DbConnect = _ConnectToDatabase(details)
        cmd = _ConstructSQL(details, extraFields=details.fpColName)
        curs = conn.GetCursor()
        if _dataSeq:
            suppl = _dataSeq(curs, cmd, depickle=not details.noPickle, klass=DataStructs.ExplicitBitVect)
            _dataSeq._conn = conn
            return suppl
        return DbFpSupplier.ForwardDbFpSupplier(data, fpColName=details.fpColName)
    if details.inFileName:
        try:
            inF = open(details.inFileName, 'r')
        except IOError:
            import traceback
            FingerprintMols.error(f'Error: Problems reading from file {details.inFileName}\n')
            traceback.print_exc()
        suppl = []
        done = 0
        while not done:
            try:
                ID, fp = pickle.load(inF)
            except Exception:
                done = 1
            else:
                fp._fieldsFromDb = [ID]
                suppl.append(fp)
        return suppl
    return None
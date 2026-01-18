import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import DbFpSupplier, FingerprintMols
from rdkit.DataStructs.TopNContainer import TopNContainer
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def _ConnectToDatabase(details) -> DbConnect:
    if details.dbName and details.tableName:
        try:
            conn = DbConnect(details.dbName, details.tableName)
            if hasattr(details, 'dbUser'):
                conn.user = details.dbUser
            if hasattr(details, 'dbPassword'):
                conn.password = details.dbPassword
            return conn
        except Exception:
            import traceback
            FingerprintMols.error(f'Error: Problems establishing connection to database:{details.dbName}|{details.tableName}\n')
            traceback.print_exc()
    return None
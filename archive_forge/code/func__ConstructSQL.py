import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import DbFpSupplier, FingerprintMols
from rdkit.DataStructs.TopNContainer import TopNContainer
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def _ConstructSQL(details, extraFields=''):
    fields = f'{details.tableName}.{details.idName}'
    join = ''
    if details.smilesTableName:
        if details.smilesName:
            fields += f',{details.smilesName}'
        join = f'join {details.smilesTableName} smi on smi.{details.idName}={details.tableName}.{details.idName}'
    if details.actTableName:
        if details.actName:
            fields += f',{details.actName}'
        join += f'join {details.actTableName} act on act.{details.idName}={details.tableName}.{details.idName}'
    if extraFields:
        fields += f',{extraFields}'
    return f'select {fields} from {details.tableName} {join}'
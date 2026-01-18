import re
import rdkit.RDLogger as logging
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def LoadDb(suppl, dbName, nameProp='_Name', nameCol='compound_id', silent=False, redraw=False, errorsTo=None, keepHs=False, defaultVal='N/A', skipProps=False, regName='molecules', skipSmiles=False, maxRowsCached=-1, uniqNames=False, addComputedProps=False, lazySupplier=False, startAnew=True):
    if not lazySupplier:
        nMols = len(suppl)
    else:
        nMols = -1
    if not silent:
        logger.info(f'Generating molecular database in file {dbName}')
        if not lazySupplier:
            logger.info(f'  Processing {nMols} molecules')
    rows = []
    globalProps = {}
    namesSeen = set()
    nDone = 0
    typeConversions = {0: ('varchar', str), 1: ('float', float), 2: ('int', int)}
    for m in suppl:
        nDone += 1
        if not m:
            if errorsTo:
                if hasattr(suppl, 'GetItemText'):
                    d = suppl.GetItemText(nDone - 1)
                    errorsTo.write(d)
                else:
                    logger.warning('full error file support not complete')
            continue
        row = ProcessMol(m, typeConversions, globalProps, nDone, nameProp=nameProp, nameCol=nameCol, redraw=redraw, keepHs=keepHs, skipProps=skipProps, addComputedProps=addComputedProps, skipSmiles=skipSmiles, uniqNames=uniqNames, namesSeen=namesSeen)
        if row is None:
            continue
        rows.append([nDone] + row)
        if not silent and (not nDone % 100):
            logger.info(f'  done {nDone}')
        if len(rows) == maxRowsCached:
            break
    nameDef = f'{nameCol} varchar not null'
    if uniqNames:
        nameDef += ' unique'
    typs = ['guid integer not null primary key', nameDef]
    pns = []
    for pn, v in globalProps.items():
        addNm = re.sub('[\\W]', '_', pn)
        typs.append(f'{addNm} {typeConversions[v][0]}')
        pns.append(pn.lower())
    if not skipSmiles:
        if 'smiles' not in pns:
            typs.append('smiles varchar')
        else:
            typs.append('cansmiles varchar')
    typs.append(f'molpkl {DbModule.binaryTypeName}')
    conn = DbConnect(dbName)
    curs = conn.GetCursor()
    if startAnew:
        try:
            curs.execute(f'drop table {regName}')
        except Exception:
            pass
        curs.execute(f'create table {regName} ({','.join(typs)})')
    else:
        curs.execute(f'select * from {regName} limit 1')
        ocolns = set([x[0] for x in curs.description])
        ncolns = set([x.split()[0] for x in typs])
        if ncolns != ocolns:
            raise ValueError(f'Column names do not match: {ocolns} != {ncolns}')
        curs.execute(f'select max(guid) from {regName}')
        offset = curs.fetchone()[0]
        for row in rows:
            row[0] += offset
    qs = ','.join([DbModule.placeHolder for _ in typs])
    ConvertRows(rows, globalProps, defaultVal, skipSmiles)
    curs.executemany(f'insert into {regName} values ({qs})', rows)
    conn.Commit()
    rows = []
    while 1:
        nDone += 1
        try:
            m = next(suppl)
        except StopIteration:
            break
        if not m:
            if errorsTo:
                if hasattr(suppl, 'GetItemText'):
                    d = suppl.GetItemText(nDone - 1)
                    errorsTo.write(d)
                else:
                    logger.warning('full error file support not complete')
            continue
        row = ProcessMol(m, typeConversions, globalProps, nDone, nameProp=nameProp, nameCol=nameCol, redraw=redraw, keepHs=keepHs, skipProps=skipProps, addComputedProps=addComputedProps, skipSmiles=skipSmiles, uniqNames=uniqNames, namesSeen=namesSeen)
        if not row:
            continue
        rows.append([nDone] + row)
        if not silent and (not nDone % 100):
            logger.info(f'  done {nDone}')
        if len(rows) == maxRowsCached:
            ConvertRows(rows, globalProps, defaultVal, skipSmiles)
            curs.executemany(f'insert into {regName} values ({qs})', rows)
            conn.Commit()
            rows = []
    if len(rows):
        ConvertRows(rows, globalProps, defaultVal, skipSmiles)
        curs.executemany(f'insert into {regName} values ({qs})', rows)
        conn.Commit()
import os
import re
from rdkit import RDConfig
def GetAtomicData(atomDict, descriptorsDesired, dBase=_atomDbName, table='atomic_data', where='', user='sysdba', password='masterkey', includeElCounts=0):
    """ pulls atomic data from a database

      **Arguments**

        - atomDict: the dictionary to populate

        - descriptorsDesired: the descriptors to pull for each atom

        - dBase: the DB to use

        - table: the DB table to use

        - where: the SQL where clause

        - user: the user name to use with the DB

        - password: the password to use with the DB

        - includeElCounts: if nonzero, valence electron count fields are added to
           the _atomDict_

    """
    extraFields = ['NVAL', 'NVAL_NO_FULL_F', 'NVAL_NO_FULL_D', 'NVAL_NO_FULL']
    from rdkit.Dbase import DbModule
    cn = DbModule.connect(dBase, user, password)
    c = cn.cursor()
    descriptorsDesired = [s.upper() for s in descriptorsDesired]
    if 'NAME' not in descriptorsDesired:
        descriptorsDesired.append('NAME')
    if includeElCounts and 'CONFIG' not in descriptorsDesired:
        descriptorsDesired.append('CONFIG')
    for field in extraFields:
        if field in descriptorsDesired:
            descriptorsDesired.remove(field)
    toPull = ','.join(descriptorsDesired)
    command = 'select %s from atomic_data %s' % (toPull, where)
    try:
        c.execute(command)
    except Exception:
        print('Problems executing command:', command)
        return
    res = c.fetchall()
    for atom in res:
        tDict = {}
        for i in range(len(descriptorsDesired)):
            desc = descriptorsDesired[i]
            val = atom[i]
            tDict[desc] = val
        name = tDict['NAME']
        atomDict[name] = tDict
        if includeElCounts:
            config = atomDict[name]['CONFIG']
            atomDict[name]['NVAL'] = ConfigToNumElectrons(config)
            atomDict[name]['NVAL_NO_FULL_F'] = ConfigToNumElectrons(config, ignoreFullF=1)
            atomDict[name]['NVAL_NO_FULL_D'] = ConfigToNumElectrons(config, ignoreFullD=1)
            atomDict[name]['NVAL_NO_FULL'] = ConfigToNumElectrons(config, ignoreFullF=1, ignoreFullD=1)
import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_structdesc(f):
    """Function to read in a structure descriptor"""
    structdesc = {}
    structstart = _read_long(f)
    if structstart != 9:
        raise Exception('STRUCTSTART should be 9')
    structdesc['name'] = _read_string(f)
    predef = _read_long(f)
    structdesc['ntags'] = _read_long(f)
    structdesc['nbytes'] = _read_long(f)
    structdesc['predef'] = predef & 1
    structdesc['inherits'] = predef & 2
    structdesc['is_super'] = predef & 4
    if not structdesc['predef']:
        structdesc['tagtable'] = [_read_tagdesc(f) for _ in range(structdesc['ntags'])]
        for tag in structdesc['tagtable']:
            tag['name'] = _read_string(f)
        structdesc['arrtable'] = {tag['name']: _read_arraydesc(f) for tag in structdesc['tagtable'] if tag['array']}
        structdesc['structtable'] = {tag['name']: _read_structdesc(f) for tag in structdesc['tagtable'] if tag['structure']}
        if structdesc['inherits'] or structdesc['is_super']:
            structdesc['classname'] = _read_string(f)
            structdesc['nsupclasses'] = _read_long(f)
            structdesc['supclassnames'] = [_read_string(f) for _ in range(structdesc['nsupclasses'])]
            structdesc['supclasstable'] = [_read_structdesc(f) for _ in range(structdesc['nsupclasses'])]
        STRUCT_DICT[structdesc['name']] = structdesc
    else:
        if structdesc['name'] not in STRUCT_DICT:
            raise Exception("PREDEF=1 but can't find definition")
        structdesc = STRUCT_DICT[structdesc['name']]
    return structdesc
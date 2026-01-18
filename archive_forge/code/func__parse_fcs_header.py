from .. import utils
from .._lazyload import fcsparser
from .utils import _matrix_to_data_frame
from io import BytesIO
import numpy as np
import pandas as pd
import string
import struct
import warnings
def _parse_fcs_header(meta):
    if meta['__header__']['data start'] == 0 and meta['__header__']['data end'] == 0:
        meta['$DATASTART'] = int(meta['$DATASTART'])
        meta['$DATAEND'] = int(meta['$DATAEND'])
    else:
        meta['$DATASTART'] = meta['__header__']['data start']
        meta['$DATAEND'] = meta['__header__']['data end']
    meta['$PAR'] = int(meta['$PAR'])
    meta['$TOT'] = int(meta['$TOT'])
    meta['$DATATYPE'] = meta['$DATATYPE'].lower()
    if meta['$DATATYPE'] not in ['f', 'd']:
        raise ValueError("Expected $DATATYPE in ['F', 'D']. Got '{}'".format(meta['$DATATYPE']))
    endian = meta['$BYTEORD']
    if endian == '4,3,2,1':
        meta['$ENDIAN'] = '>'
    elif endian == '1,2,3,4':
        meta['$ENDIAN'] = '<'
    else:
        raise ValueError("Expected $BYTEORD in ['1,2,3,4', '4,3,2,1']. Got '{}'".format(endian))
    return meta
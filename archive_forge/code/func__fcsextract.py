from .. import utils
from .._lazyload import fcsparser
from .utils import _matrix_to_data_frame
from io import BytesIO
import numpy as np
import pandas as pd
import string
import struct
import warnings
def _fcsextract(filename, channel_naming='$PnS', reformat_meta=True):
    """Parse FCS with experimental parser.

    Some files fail to load with `fcsparser.parse`. For these, we provide an
    alternative parser. It is not guaranteed to work in all cases.

    Code copied from https://github.com/pontikos/fcstools/blob/master/fcs.extract.py

    Paramseters
    -----------
    channel_naming: '$PnS' | '$PnN'
        Determines which meta data field is used for naming the channels.
        The default should be $PnS (even though it is not guaranteed to be unique)
        $PnN stands for the short name (guaranteed to be unique).
            Will look like 'FL1-H'
        $PnS stands for the actual name (not guaranteed to be unique).
            Will look like 'FSC-H' (Forward scatter)
        The chosen field will be used to population self.channels
        Note: These names are not flipped in the implementation.
        It looks like they were swapped for some reason in the official FCS
        specification.
    reformat_meta: bool
        If true, the meta data is reformatted with the channel information organized
        into a DataFrame and moved into the '_channels_' key
    """
    meta = _read_fcs_header(filename)
    meta = _parse_fcs_header(meta)
    with open(filename, 'rb') as handle:
        handle.seek(meta['$DATASTART'])
        data = handle.read(meta['$DATAEND'] - meta['$DATASTART'] + 1)
        data = BytesIO(data)
        fmt = meta['$ENDIAN'] + str(meta['$PAR']) + meta['$DATATYPE']
        datasize = struct.calcsize(fmt)
        events = []
        for e in range(meta['$TOT']):
            event = struct.unpack(fmt, data.read(datasize))
            events.append(event)
    pars = meta['$PAR']
    if '$P0B' in meta:
        channel_numbers = range(0, pars)
    else:
        channel_numbers = range(1, pars + 1)
    channel_names = _get_channel_names(meta, channel_numbers, channel_naming)
    events = pd.DataFrame(np.array(events), columns=channel_names, index=np.arange(len(events)))
    if reformat_meta:
        try:
            meta['_channels_'] = _reformat_meta(meta, channel_numbers)
        except Exception as exp:
            warnings.warn('Metadata reformatting failed: {}'.format(str(exp)))
        meta['_channel_names_'] = channel_names
    return (meta, events)
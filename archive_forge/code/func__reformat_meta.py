from .. import utils
from .._lazyload import fcsparser
from .utils import _matrix_to_data_frame
from io import BytesIO
import numpy as np
import pandas as pd
import string
import struct
import warnings
def _reformat_meta(meta, channel_numbers):
    """Collect the meta data information in a more user friendly format.

    Looks through the meta data, collecting the channel related information
    into a dataframe and moving it into the _channels_ key.

    Credit: https://github.com/eyurtsev/fcsparser/blob/master/fcsparser/api.py
    """
    channel_properties = []
    for key, value in meta.items():
        if key[:3] == '$P1':
            if key[3] not in string.digits:
                channel_properties.append(key[3:])
    channel_matrix = [[meta.get('$P{0}{1}'.format(ch, p)) for p in channel_properties] for ch in channel_numbers]
    for ch in channel_numbers:
        for p in channel_properties:
            key = '$P{0}{1}'.format(ch, p)
            if key in meta:
                meta.pop(key)
    num_channels = meta['$PAR']
    column_names = ['$Pn{0}'.format(p) for p in channel_properties]
    df = pd.DataFrame(channel_matrix, columns=column_names, index=1 + np.arange(num_channels))
    if '$PnE' in column_names:
        df['$PnE'] = df['$PnE'].apply(lambda x: x.split(','))
    if '$PnB' in column_names:
        df['$PnB'] = df['$PnB'].apply(lambda x: int(x))
    df.index.name = 'Channel Number'
    return df
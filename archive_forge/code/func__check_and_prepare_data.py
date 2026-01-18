import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _check_and_prepare_data(data, config):
    """
    Check and Prepare the data input:
    For now, data must be a Pandas DataFrame with a DatetimeIndex
    and columns named 'Open', 'High', 'Low', 'Close', and optionally 'Volume'

    Later (if there is demand for it) we may accept all of the following data formats:
      1. Pandas DataFrame with DatetimeIndex (as described above)
      2. Pandas Series with DatetimeIndex:
             Values are close prices, and Series generates a line plot
      3. Tuple of Lists, or List of Lists:
             The inner Lists are each columns, in the order: DateTime, Open, High, Low, Close, Volume
      4. Tuple of Tuples or List of Tuples:
             The inner tuples are each row, containing values in the order: DateTime, Open, High, Low, Close, Volume

    Return a Tuple of Lists: datetimes, opens, highs, lows, closes, volumes
    """
    if not isinstance(data, pd.core.frame.DataFrame):
        raise TypeError('Expect data as DataFrame')
    if not isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise TypeError('Expect data.index as DatetimeIndex')
    columns = config['columns']
    if columns is None:
        columns = ('Open', 'High', 'Low', 'Close', 'Volume')
        if all([c.lower() in data for c in columns[0:4]]):
            columns = ('open', 'high', 'low', 'close', 'volume')
    o, h, l, c, v = columns
    cols = [o, h, l, c]
    if config['volume'] != False:
        expect_cols = columns
    else:
        expect_cols = cols
    for col in expect_cols:
        if col not in data.columns:
            for dc in data.columns:
                if dc.strip() != dc:
                    warnings.warn('\n ================================================================= ' + '\n   Input DataFrame column name "' + dc + '" ' + '\n   contains leading and/or trailing whitespace.', category=UserWarning)
            raise ValueError('Column "' + col + '" NOT FOUND in Input DataFrame!' + '\n            CHECK that your column names are correct AND/OR' + '\n            CHECK for leading or trailing blanks in your column names.')
    opens = data[o].values
    highs = data[h].values
    lows = data[l].values
    closes = data[c].values
    if v in data.columns:
        volumes = data[v].values
        cols.append(v)
    else:
        volumes = None
    for col in cols:
        if not all((isinstance(v, (float, int)) for v in data[col])):
            raise ValueError('Data for column "' + str(col) + '" must be ALL float or int.')
    if config['tz_localize']:
        dates = mdates.date2num(data.index.tz_localize(None).to_pydatetime())
    else:
        dates = mdates.date2num(data.index.to_pydatetime())
    if len(data.index) > config['warn_too_much_data'] and (config['type'] == 'candle' or config['type'] == 'ohlc' or config['type'] == 'hollow_and_filled'):
        warnings.warn('\n\n ================================================================= ' + '\n\n   WARNING: YOU ARE PLOTTING SO MUCH DATA THAT IT MAY NOT BE' + '\n            POSSIBLE TO SEE DETAILS (Candles, Ohlc-Bars, Etc.)' + '\n   For more information see:' + '\n   - https://github.com/matplotlib/mplfinance/wiki/Plotting-Too-Much-Data' + '\n   ' + "\n   TO SILENCE THIS WARNING, set `type='line'` in `mpf.plot()`" + '\n   OR set kwarg `warn_too_much_data=N` where N is an integer ' + '\n   LARGER than the number of data points you want to plot.' + '\n\n ================================================================ ', category=UserWarning)
    return (dates, opens, highs, lows, closes, volumes)
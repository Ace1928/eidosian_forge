import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _fix_zeroes(self, df, interval, tz_exchange, prepost):
    if df.empty:
        return df
    logger = utils.get_yf_logger()
    if df.shape[0] == 0:
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    intraday = interval[-1] in ('m', 'h')
    df = df.sort_index()
    df2 = df.copy()
    if df2.index.tz is None:
        df2.index = df2.index.tz_localize(tz_exchange)
    elif df2.index.tz != tz_exchange:
        df2.index = df2.index.tz_convert(tz_exchange)
    price_cols = [c for c in _PRICE_COLNAMES_ if c in df2.columns]
    f_prices_bad = (df2[price_cols] == 0.0) | df2[price_cols].isna()
    df2_reserve = None
    if intraday:
        grp = pd.Series(f_prices_bad.any(axis=1), name='nan').groupby(f_prices_bad.index.date)
        nan_pct = grp.sum() / grp.count()
        dts = nan_pct.index[nan_pct > 0.5]
        f_zero_or_nan_ignore = np.isin(f_prices_bad.index.date, dts)
        df2_reserve = df2[f_zero_or_nan_ignore]
        df2 = df2[~f_zero_or_nan_ignore]
        f_prices_bad = (df2[price_cols] == 0.0) | df2[price_cols].isna()
    f_high_low_good = ~df2['High'].isna().to_numpy() & ~df2['Low'].isna().to_numpy()
    f_change = df2['High'].to_numpy() != df2['Low'].to_numpy()
    f_vol_bad = (df2['Volume'] == 0).to_numpy() & f_high_low_good & f_change
    if 'Stock Splits' in df2.columns:
        f_split = (df2['Stock Splits'] != 0.0).to_numpy()
        if f_split.any():
            f_change_expected_but_missing = f_split & ~f_change
            if f_change_expected_but_missing.any():
                f_prices_bad[f_change_expected_but_missing] = True
    f_prices_bad = f_prices_bad.to_numpy()
    f_bad_rows = f_prices_bad.any(axis=1) | f_vol_bad
    if not f_bad_rows.any():
        logger.info('price-repair-missing: No price=0 errors to repair')
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    if f_prices_bad.sum() == len(price_cols) * len(df2):
        logger.info('price-repair-missing: No good data for calibration so cannot fix price=0 bad data')
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    data_cols = price_cols + ['Volume']
    tag = -1.0
    for i in range(len(price_cols)):
        c = price_cols[i]
        df2.loc[f_prices_bad[:, i], c] = tag
    df2.loc[f_vol_bad, 'Volume'] = tag
    f_vol_zero_or_nan = (df2['Volume'].to_numpy() == 0) | df2['Volume'].isna().to_numpy()
    df2.loc[f_prices_bad.any(axis=1) & f_vol_zero_or_nan, 'Volume'] = tag
    df2.loc[f_change & f_vol_zero_or_nan, 'Volume'] = tag
    df2_tagged = df2[data_cols].to_numpy() == tag
    n_before = df2_tagged.sum()
    dts_tagged = df2.index[df2_tagged.any(axis=1)]
    df2 = self._reconstruct_intervals_batch(df2, interval, prepost, tag)
    df2_tagged = df2[data_cols].to_numpy() == tag
    n_after = df2_tagged.sum()
    dts_not_repaired = df2.index[df2_tagged.any(axis=1)]
    n_fixed = n_before - n_after
    if n_fixed > 0:
        msg = f'{self.ticker}: fixed {n_fixed}/{n_before} value=0 errors in {interval} price data'
        if n_fixed < 4:
            dts_repaired = sorted(list(set(dts_tagged).difference(dts_not_repaired)))
            msg += f': {dts_repaired}'
        logger.info('price-repair-missing: ' + msg)
    if df2_reserve is not None:
        if 'Repaired?' not in df2_reserve.columns:
            df2_reserve['Repaired?'] = False
        df2 = pd.concat([df2, df2_reserve]).sort_index()
    f = df2[data_cols].to_numpy() == tag
    for j in range(len(data_cols)):
        fj = f[:, j]
        if fj.any():
            c = data_cols[j]
            df2.loc[fj, c] = df.loc[fj, c]
    return df2
import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _reconstruct_intervals_batch(self, df, interval, prepost, tag=-1):
    logger = utils.get_yf_logger()
    if not isinstance(df, pd.DataFrame):
        raise Exception("'df' must be a Pandas DataFrame not", type(df))
    if interval == '1m':
        return df
    if interval[1:] in ['d', 'wk', 'mo']:
        prepost = True
        intraday = False
    else:
        intraday = True
    price_cols = [c for c in _PRICE_COLNAMES_ if c in df]
    data_cols = price_cols + ['Volume']
    intervals = ['1wk', '1d', '1h', '30m', '15m', '5m', '2m', '1m']
    itds = {i: utils._interval_to_timedelta(interval) for i in intervals}
    nexts = {intervals[i]: intervals[i + 1] for i in range(len(intervals) - 1)}
    min_lookbacks = {'1wk': None, '1d': None, '1h': _datetime.timedelta(days=730)}
    for i in ['30m', '15m', '5m', '2m']:
        min_lookbacks[i] = _datetime.timedelta(days=60)
    min_lookbacks['1m'] = _datetime.timedelta(days=30)
    if interval in nexts:
        sub_interval = nexts[interval]
        td_range = itds[interval]
    else:
        logger.warning(f"Have not implemented price repair for '{interval}' interval. Contact developers")
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    if self._reconstruct_start_interval is None:
        self._reconstruct_start_interval = interval
    if interval != self._reconstruct_start_interval and interval != nexts[self._reconstruct_start_interval]:
        logger.debug(f"{self.ticker}: Price repair has hit max depth of 2 ('%s'->'%s'->'%s')", self._reconstruct_start_interval, nexts[self._reconstruct_start_interval], interval)
        return df
    df = df.sort_index()
    f_repair = df[data_cols].to_numpy() == tag
    f_repair_rows = f_repair.any(axis=1)
    m = min_lookbacks[sub_interval]
    if m is None:
        min_dt = None
    else:
        m -= _datetime.timedelta(days=1)
        min_dt = pd.Timestamp.utcnow() - m
        min_dt = min_dt.tz_convert(df.index.tz).ceil('D')
    logger.debug(f'min_dt={min_dt} interval={interval} sub_interval={sub_interval}')
    if min_dt is not None:
        f_recent = df.index >= min_dt
        f_repair_rows = f_repair_rows & f_recent
        if not f_repair_rows.any():
            logger.info('Data too old to repair')
            if 'Repaired?' not in df.columns:
                df['Repaired?'] = False
            return df
    dts_to_repair = df.index[f_repair_rows]
    if len(dts_to_repair) == 0:
        logger.info('Nothing needs repairing (dts_to_repair[] empty)')
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    df_v2 = df.copy()
    if 'Repaired?' not in df_v2.columns:
        df_v2['Repaired?'] = False
    f_good = ~df[price_cols].isna().any(axis=1)
    f_good = f_good & (df[price_cols].to_numpy() != tag).all(axis=1)
    df_good = df[f_good]
    dts_groups = [[dts_to_repair[0]]]
    if sub_interval == '1mo':
        grp_max_size = _dateutil.relativedelta.relativedelta(years=2)
    elif sub_interval == '1wk':
        grp_max_size = _dateutil.relativedelta.relativedelta(years=2)
    elif sub_interval == '1d':
        grp_max_size = _dateutil.relativedelta.relativedelta(years=2)
    elif sub_interval == '1h':
        grp_max_size = _dateutil.relativedelta.relativedelta(years=1)
    elif sub_interval == '1m':
        grp_max_size = _datetime.timedelta(days=5)
    else:
        grp_max_size = _datetime.timedelta(days=30)
    logger.debug(f'grp_max_size = {grp_max_size}')
    for i in range(1, len(dts_to_repair)):
        dt = dts_to_repair[i]
        if dt.date() < dts_groups[-1][0].date() + grp_max_size:
            dts_groups[-1].append(dt)
        else:
            dts_groups.append([dt])
    logger.debug('Repair groups:')
    for g in dts_groups:
        logger.debug(f'- {g[0]} -> {g[-1]}')
    for i in range(len(dts_groups)):
        g = dts_groups[i]
        g0 = g[0]
        i0 = df_good.index.get_indexer([g0], method='nearest')[0]
        if i0 > 0:
            if (min_dt is None or df_good.index[i0 - 1] >= min_dt) and (not intraday or df_good.index[i0 - 1].date() == g0.date()):
                i0 -= 1
        gl = g[-1]
        il = df_good.index.get_indexer([gl], method='nearest')[0]
        if il < len(df_good) - 1:
            if not intraday or df_good.index[il + 1].date() == gl.date():
                il += 1
        good_dts = df_good.index[i0:il + 1]
        dts_groups[i] += good_dts.to_list()
        dts_groups[i].sort()
    n_fixed = 0
    for g in dts_groups:
        df_block = df[df.index.isin(g)]
        logger.debug('df_block:\n' + str(df_block))
        start_dt = g[0]
        start_d = start_dt.date()
        reject = False
        if sub_interval == '1h' and _datetime.date.today() - start_d > _datetime.timedelta(days=729):
            reject = True
        elif sub_interval in ['30m', '15m'] and _datetime.date.today() - start_d > _datetime.timedelta(days=59):
            reject = True
        if reject:
            msg = f'Cannot reconstruct {interval} block starting'
            if intraday:
                msg += f' {start_dt}'
            else:
                msg += f' {start_d}'
            msg += ', too old, Yahoo will reject request for finer-grain data'
            logger.info(msg)
            continue
        td_1d = _datetime.timedelta(days=1)
        end_dt = g[-1]
        end_d = end_dt.date() + td_1d
        if interval in '1wk':
            fetch_start = start_d - td_range
            fetch_end = g[-1].date() + td_range
        elif interval == '1d':
            fetch_start = start_d
            fetch_end = g[-1].date() + td_range
        else:
            fetch_start = g[0]
            fetch_end = g[-1] + td_range
        fetch_start -= td_1d
        fetch_end += td_1d
        if intraday:
            fetch_start = fetch_start.date()
            fetch_end = fetch_end.date() + td_1d
        if min_dt is not None:
            fetch_start = max(min_dt.date(), fetch_start)
        logger.debug(f'Fetching {sub_interval} prepost={prepost} {fetch_start}->{fetch_end}')
        df_fine = self.history(start=fetch_start, end=fetch_end, interval=sub_interval, auto_adjust=False, actions=True, prepost=prepost, repair=True, keepna=True)
        if df_fine is None or df_fine.empty:
            msg = f'Cannot reconstruct {interval} block starting'
            if intraday:
                msg += f' {start_dt}'
            else:
                msg += f' {start_d}'
            msg += ', too old, Yahoo is rejecting request for finer-grain data'
            logger.debug(msg)
            continue
        df_fine = df_fine.loc[g[0]:g[-1] + itds[sub_interval] - _datetime.timedelta(milliseconds=1)].copy()
        if df_fine.empty:
            msg = f'Cannot reconstruct {interval} block range'
            if intraday:
                msg += f' {start_dt}->{end_dt}'
            else:
                msg += f' {start_d}->{end_d}'
            msg += ', Yahoo not returning finer-grain data within range'
            logger.debug(msg)
            continue
        df_fine['ctr'] = 0
        if interval == '1wk':
            weekdays = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
            week_end_day = weekdays[(df_block.index[0].weekday() + 7 - 1) % 7]
            df_fine['Week Start'] = df_fine.index.tz_localize(None).to_period('W-' + week_end_day).start_time
            grp_col = 'Week Start'
        elif interval == '1d':
            df_fine['Day Start'] = pd.to_datetime(df_fine.index.date)
            grp_col = 'Day Start'
        else:
            df_fine.loc[df_fine.index.isin(df_block.index), 'ctr'] = 1
            df_fine['intervalID'] = df_fine['ctr'].cumsum()
            df_fine = df_fine.drop('ctr', axis=1)
            grp_col = 'intervalID'
        df_fine = df_fine[~df_fine[price_cols + ['Dividends']].isna().all(axis=1)]
        df_fine_grp = df_fine.groupby(grp_col)
        df_new = df_fine_grp.agg(Open=('Open', 'first'), Close=('Close', 'last'), AdjClose=('Adj Close', 'last'), Low=('Low', 'min'), High=('High', 'max'), Dividends=('Dividends', 'sum'), Volume=('Volume', 'sum')).rename(columns={'AdjClose': 'Adj Close'})
        if grp_col in ['Week Start', 'Day Start']:
            df_new.index = df_new.index.tz_localize(df_fine.index.tz)
        else:
            df_fine['diff'] = df_fine['intervalID'].diff()
            new_index = np.append([df_fine.index[0]], df_fine.index[df_fine['intervalID'].diff() > 0])
            df_new.index = new_index
        logger.debug('df_new:' + '\n' + str(df_new))
        common_index = np.intersect1d(df_block.index, df_new.index)
        if len(common_index) == 0:
            logger.info(f"Can't calibrate {interval} block starting {start_d} so aborting repair")
            continue
        if interval == '1d':
            df_new_calib = df_new[df_new.index.isin(common_index)]
            df_block_calib = df_block[df_block.index.isin(common_index)]
            f_tag = df_block_calib['Adj Close'] == tag
            if f_tag.any():
                div_adjusts = df_block_calib['Adj Close'] / df_block_calib['Close']
                div_adjusts[f_tag] = np.nan
                div_adjusts = div_adjusts.ffill().bfill()
                for idx in np.where(f_tag)[0]:
                    dt = df_new_calib.index[idx]
                    n = len(div_adjusts)
                    if df_new.loc[dt, 'Dividends'] != 0:
                        if idx < n - 1:
                            div_adjusts.iloc[idx] = div_adjusts.iloc[idx + 1]
                        else:
                            div_adj = 1.0 - df_new_calib['Dividends'].iloc[idx] / df_new_calib['Close'].iloc[idx - 1]
                            div_adjusts.iloc[idx] = div_adjusts.iloc[idx - 1] / div_adj
                    elif idx > 0:
                        div_adjusts.iloc[idx] = div_adjusts.iloc[idx - 1]
                    else:
                        div_adjusts.iloc[idx] = div_adjusts.iloc[idx + 1]
                        if df_new_calib['Dividends'].iloc[idx + 1] != 0:
                            div_adjusts.iloc[idx] *= 1.0 - df_new_calib['Dividends'].iloc[idx + 1] / df_new_calib['Close'].iloc[idx]
                f_close_bad = df_block_calib['Close'] == tag
                div_adjusts = div_adjusts.reindex(df_block.index, fill_value=np.nan).ffill().bfill()
                df_new['Adj Close'] = df_block['Close'] * div_adjusts
                if f_close_bad.any():
                    f_close_bad_new = f_close_bad.reindex(df_new.index, fill_value=False)
                    div_adjusts_new = div_adjusts.reindex(df_new.index, fill_value=np.nan).ffill().bfill()
                    div_adjusts_new_np = f_close_bad_new.to_numpy()
                    df_new.loc[div_adjusts_new_np, 'Adj Close'] = df_new['Close'][div_adjusts_new_np] * div_adjusts_new[div_adjusts_new_np]
        calib_cols = ['Open', 'Close']
        df_new_calib = df_new[df_new.index.isin(common_index)][calib_cols].to_numpy()
        df_block_calib = df_block[df_block.index.isin(common_index)][calib_cols].to_numpy()
        calib_filter = df_block_calib != tag
        if not calib_filter.any():
            logger.info(f"Can't calibrate {interval} block starting {start_d} so aborting repair")
            continue
        for j in range(len(calib_cols)):
            f = ~calib_filter[:, j]
            if f.any():
                df_block_calib[f, j] = 1
                df_new_calib[f, j] = 1
        ratios = df_block_calib[calib_filter] / df_new_calib[calib_filter]
        weights = df_fine_grp.size()
        weights.index = df_new.index
        weights = weights[weights.index.isin(common_index)].to_numpy().astype(float)
        weights = weights[:, None]
        weights = np.tile(weights, len(calib_cols))
        weights = weights[calib_filter]
        not1 = ~np.isclose(ratios, 1.0, rtol=1e-05)
        if np.sum(not1) == len(calib_cols):
            ratio = 1.0
        else:
            ratio = np.average(ratios, weights=weights)
        logger.debug(f'Price calibration ratio (raw) = {ratio:6f}')
        ratio_rcp = round(1.0 / ratio, 1)
        ratio = round(ratio, 1)
        if ratio == 1 and ratio_rcp == 1:
            pass
        elif ratio > 1:
            df_new[price_cols] *= ratio
            df_new['Volume'] /= ratio
        elif ratio_rcp > 1:
            df_new[price_cols] *= 1.0 / ratio_rcp
            df_new['Volume'] *= ratio_rcp
        bad_dts = df_block.index[(df_block[price_cols + ['Volume']] == tag).to_numpy().any(axis=1)]
        no_fine_data_dts = []
        for idx in bad_dts:
            if idx not in df_new.index:
                no_fine_data_dts.append(idx)
        if len(no_fine_data_dts) > 0:
            logger.debug("Yahoo didn't return finer-grain data for these intervals: " + str(no_fine_data_dts))
        for idx in bad_dts:
            if idx not in df_new.index:
                continue
            df_new_row = df_new.loc[idx]
            if interval == '1wk':
                df_last_week = df_new.iloc[df_new.index.get_loc(idx) - 1]
                df_fine = df_fine.loc[idx:]
            df_bad_row = df.loc[idx]
            bad_fields = df_bad_row.index[df_bad_row == tag].to_numpy()
            if 'High' in bad_fields:
                df_v2.loc[idx, 'High'] = df_new_row['High']
            if 'Low' in bad_fields:
                df_v2.loc[idx, 'Low'] = df_new_row['Low']
            if 'Open' in bad_fields:
                if interval == '1wk' and idx != df_fine.index[0]:
                    df_v2.loc[idx, 'Open'] = df_last_week['Close']
                    df_v2.loc[idx, 'Low'] = min(df_v2.loc[idx, 'Open'], df_v2.loc[idx, 'Low'])
                else:
                    df_v2.loc[idx, 'Open'] = df_new_row['Open']
            if 'Close' in bad_fields:
                df_v2.loc[idx, 'Close'] = df_new_row['Close']
                df_v2.loc[idx, 'Adj Close'] = df_new_row['Adj Close']
            elif 'Adj Close' in bad_fields:
                df_v2.loc[idx, 'Adj Close'] = df_new_row['Adj Close']
            if 'Volume' in bad_fields:
                df_v2.loc[idx, 'Volume'] = df_new_row['Volume']
            df_v2.loc[idx, 'Repaired?'] = True
            n_fixed += 1
    return df_v2
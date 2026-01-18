import datetime
import json
import warnings
from collections.abc import MutableMapping
import numpy as _np
import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import quote_summary_valid_modules, _BASE_URL_
from yfinance.exceptions import YFNotImplementedError, YFinanceDataException, YFinanceException
class FastInfo:

    def __init__(self, tickerBaseObject, proxy=None):
        self._tkr = tickerBaseObject
        self.proxy = proxy
        self._prices_1y = None
        self._prices_1wk_1h_prepost = None
        self._prices_1wk_1h_reg = None
        self._md = None
        self._currency = None
        self._quote_type = None
        self._exchange = None
        self._timezone = None
        self._shares = None
        self._mcap = None
        self._open = None
        self._day_high = None
        self._day_low = None
        self._last_price = None
        self._last_volume = None
        self._prev_close = None
        self._reg_prev_close = None
        self._50d_day_average = None
        self._200d_day_average = None
        self._year_high = None
        self._year_low = None
        self._year_change = None
        self._10d_avg_vol = None
        self._3mo_avg_vol = None
        _properties = ['currency', 'quote_type', 'exchange', 'timezone']
        _properties += ['shares', 'market_cap']
        _properties += ['last_price', 'previous_close', 'open', 'day_high', 'day_low']
        _properties += ['regular_market_previous_close']
        _properties += ['last_volume']
        _properties += ['fifty_day_average', 'two_hundred_day_average', 'ten_day_average_volume', 'three_month_average_volume']
        _properties += ['year_high', 'year_low', 'year_change']
        base_keys = [k for k in _properties if '_' not in k]
        sc_keys = [k for k in _properties if '_' in k]
        self._sc_to_cc_key = {k: utils.snake_case_2_camelCase(k) for k in sc_keys}
        self._cc_to_sc_key = {v: k for k, v in self._sc_to_cc_key.items()}
        self._public_keys = sorted(base_keys + list(self._sc_to_cc_key.values()))
        self._keys = sorted(self._public_keys + sc_keys)

    def keys(self):
        return self._public_keys

    def items(self):
        return [(k, self[k]) for k in self._public_keys]

    def values(self):
        return [self[k] for k in self._public_keys]

    def get(self, key, default=None):
        if key in self.keys():
            if key in self._cc_to_sc_key:
                key = self._cc_to_sc_key[key]
            return self[key]
        return default

    def __getitem__(self, k):
        if not isinstance(k, str):
            raise KeyError('key must be a string')
        if k not in self._keys:
            raise KeyError(f"'{k}' not valid key. Examine 'FastInfo.keys()'")
        if k in self._cc_to_sc_key:
            k = self._cc_to_sc_key[k]
        return getattr(self, k)

    def __contains__(self, k):
        return k in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def __str__(self):
        return 'lazy-loading dict with keys = ' + str(self.keys())

    def __repr__(self):
        return self.__str__()

    def toJSON(self, indent=4):
        return json.dumps({k: self[k] for k in self.keys()}, indent=indent)

    def _get_1y_prices(self, fullDaysOnly=False):
        if self._prices_1y is None:
            self._prices_1y = self._tkr.history(period='380d', auto_adjust=False, keepna=True, proxy=self.proxy)
            self._md = self._tkr.get_history_metadata(proxy=self.proxy)
            try:
                ctp = self._md['currentTradingPeriod']
                self._today_open = pd.to_datetime(ctp['regular']['start'], unit='s', utc=True).tz_convert(self.timezone)
                self._today_close = pd.to_datetime(ctp['regular']['end'], unit='s', utc=True).tz_convert(self.timezone)
                self._today_midnight = self._today_close.ceil('D')
            except Exception:
                self._today_open = None
                self._today_close = None
                self._today_midnight = None
                raise
        if self._prices_1y.empty:
            return self._prices_1y
        dnow = pd.Timestamp.utcnow().tz_convert(self.timezone).date()
        d1 = dnow
        d0 = d1 + datetime.timedelta(days=1) - utils._interval_to_timedelta('1y')
        if fullDaysOnly and self._exchange_open_now():
            d1 -= utils._interval_to_timedelta('1d')
        return self._prices_1y.loc[str(d0):str(d1)]

    def _get_1wk_1h_prepost_prices(self):
        if self._prices_1wk_1h_prepost is None:
            self._prices_1wk_1h_prepost = self._tkr.history(period='1wk', interval='1h', auto_adjust=False, prepost=True, proxy=self.proxy)
        return self._prices_1wk_1h_prepost

    def _get_1wk_1h_reg_prices(self):
        if self._prices_1wk_1h_reg is None:
            self._prices_1wk_1h_reg = self._tkr.history(period='1wk', interval='1h', auto_adjust=False, prepost=False, proxy=self.proxy)
        return self._prices_1wk_1h_reg

    def _get_exchange_metadata(self):
        if self._md is not None:
            return self._md
        self._get_1y_prices()
        self._md = self._tkr.get_history_metadata(proxy=self.proxy)
        return self._md

    def _exchange_open_now(self):
        t = pd.Timestamp.utcnow()
        self._get_exchange_metadata()
        last_day_cutoff = self._get_1y_prices().index[-1] + datetime.timedelta(days=1)
        last_day_cutoff += datetime.timedelta(minutes=20)
        r = t < last_day_cutoff
        return r

    @property
    def currency(self):
        if self._currency is not None:
            return self._currency
        md = self._tkr.get_history_metadata(proxy=self.proxy)
        self._currency = md['currency']
        return self._currency

    @property
    def quote_type(self):
        if self._quote_type is not None:
            return self._quote_type
        md = self._tkr.get_history_metadata(proxy=self.proxy)
        self._quote_type = md['instrumentType']
        return self._quote_type

    @property
    def exchange(self):
        if self._exchange is not None:
            return self._exchange
        self._exchange = self._get_exchange_metadata()['exchangeName']
        return self._exchange

    @property
    def timezone(self):
        if self._timezone is not None:
            return self._timezone
        self._timezone = self._get_exchange_metadata()['exchangeTimezoneName']
        return self._timezone

    @property
    def shares(self):
        if self._shares is not None:
            return self._shares
        shares = self._tkr.get_shares_full(start=pd.Timestamp.utcnow().date() - pd.Timedelta(days=548), proxy=self.proxy)
        if shares is not None:
            if isinstance(shares, pd.DataFrame):
                shares = shares[shares.columns[0]]
            self._shares = int(shares.iloc[-1])
        return self._shares

    @property
    def last_price(self):
        if self._last_price is not None:
            return self._last_price
        prices = self._get_1y_prices()
        if prices.empty:
            md = self._get_exchange_metadata()
            if 'regularMarketPrice' in md:
                self._last_price = md['regularMarketPrice']
        else:
            self._last_price = float(prices['Close'].iloc[-1])
            if _np.isnan(self._last_price):
                md = self._get_exchange_metadata()
                if 'regularMarketPrice' in md:
                    self._last_price = md['regularMarketPrice']
        return self._last_price

    @property
    def previous_close(self):
        if self._prev_close is not None:
            return self._prev_close
        prices = self._get_1wk_1h_prepost_prices()
        fail = False
        if prices.empty:
            fail = True
        else:
            prices = prices[['Close']].groupby(prices.index.date).last()
            if prices.shape[0] < 2:
                fail = True
            else:
                self._prev_close = float(prices['Close'].iloc[-2])
        if fail:
            self._tkr.info
            k = 'previousClose'
            if self._tkr._quote._retired_info is not None and k in self._tkr._quote._retired_info:
                self._prev_close = self._tkr._quote._retired_info[k]
        return self._prev_close

    @property
    def regular_market_previous_close(self):
        if self._reg_prev_close is not None:
            return self._reg_prev_close
        prices = self._get_1y_prices()
        if prices.shape[0] == 1:
            prices = self._get_1wk_1h_reg_prices()
            prices = prices[['Close']].groupby(prices.index.date).last()
        if prices.shape[0] < 2:
            self._tkr.info
            k = 'regularMarketPreviousClose'
            if self._tkr._quote._retired_info is not None and k in self._tkr._quote._retired_info:
                self._reg_prev_close = self._tkr._quote._retired_info[k]
        else:
            self._reg_prev_close = float(prices['Close'].iloc[-2])
        return self._reg_prev_close

    @property
    def open(self):
        if self._open is not None:
            return self._open
        prices = self._get_1y_prices()
        if prices.empty:
            self._open = None
        else:
            self._open = float(prices['Open'].iloc[-1])
            if _np.isnan(self._open):
                self._open = None
        return self._open

    @property
    def day_high(self):
        if self._day_high is not None:
            return self._day_high
        prices = self._get_1y_prices()
        if prices.empty:
            self._day_high = None
        else:
            self._day_high = float(prices['High'].iloc[-1])
            if _np.isnan(self._day_high):
                self._day_high = None
        return self._day_high

    @property
    def day_low(self):
        if self._day_low is not None:
            return self._day_low
        prices = self._get_1y_prices()
        if prices.empty:
            self._day_low = None
        else:
            self._day_low = float(prices['Low'].iloc[-1])
            if _np.isnan(self._day_low):
                self._day_low = None
        return self._day_low

    @property
    def last_volume(self):
        if self._last_volume is not None:
            return self._last_volume
        prices = self._get_1y_prices()
        self._last_volume = None if prices.empty else int(prices['Volume'].iloc[-1])
        return self._last_volume

    @property
    def fifty_day_average(self):
        if self._50d_day_average is not None:
            return self._50d_day_average
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.empty:
            self._50d_day_average = None
        else:
            n = prices.shape[0]
            a = n - 50
            b = n
            if a < 0:
                a = 0
            self._50d_day_average = float(prices['Close'].iloc[a:b].mean())
        return self._50d_day_average

    @property
    def two_hundred_day_average(self):
        if self._200d_day_average is not None:
            return self._200d_day_average
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.empty:
            self._200d_day_average = None
        else:
            n = prices.shape[0]
            a = n - 200
            b = n
            if a < 0:
                a = 0
            self._200d_day_average = float(prices['Close'].iloc[a:b].mean())
        return self._200d_day_average

    @property
    def ten_day_average_volume(self):
        if self._10d_avg_vol is not None:
            return self._10d_avg_vol
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.empty:
            self._10d_avg_vol = None
        else:
            n = prices.shape[0]
            a = n - 10
            b = n
            if a < 0:
                a = 0
            self._10d_avg_vol = int(prices['Volume'].iloc[a:b].mean())
        return self._10d_avg_vol

    @property
    def three_month_average_volume(self):
        if self._3mo_avg_vol is not None:
            return self._3mo_avg_vol
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.empty:
            self._3mo_avg_vol = None
        else:
            dt1 = prices.index[-1]
            dt0 = dt1 - utils._interval_to_timedelta('3mo') + utils._interval_to_timedelta('1d')
            self._3mo_avg_vol = int(prices.loc[dt0:dt1, 'Volume'].mean())
        return self._3mo_avg_vol

    @property
    def year_high(self):
        if self._year_high is not None:
            return self._year_high
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.empty:
            prices = self._get_1y_prices(fullDaysOnly=False)
        self._year_high = float(prices['High'].max())
        return self._year_high

    @property
    def year_low(self):
        if self._year_low is not None:
            return self._year_low
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.empty:
            prices = self._get_1y_prices(fullDaysOnly=False)
        self._year_low = float(prices['Low'].min())
        return self._year_low

    @property
    def year_change(self):
        if self._year_change is not None:
            return self._year_change
        prices = self._get_1y_prices(fullDaysOnly=True)
        if prices.shape[0] >= 2:
            self._year_change = (prices['Close'].iloc[-1] - prices['Close'].iloc[0]) / prices['Close'].iloc[0]
            self._year_change = float(self._year_change)
        return self._year_change

    @property
    def market_cap(self):
        if self._mcap is not None:
            return self._mcap
        try:
            shares = self.shares
        except Exception as e:
            if 'Cannot retrieve share count' in str(e):
                shares = None
            elif 'failed to decrypt Yahoo' in str(e):
                shares = None
            else:
                raise
        if shares is None:
            self._tkr.info
            k = 'marketCap'
            if self._tkr._quote._retired_info is not None and k in self._tkr._quote._retired_info:
                self._mcap = self._tkr._quote._retired_info[k]
        else:
            self._mcap = float(shares * self.last_price)
        return self._mcap
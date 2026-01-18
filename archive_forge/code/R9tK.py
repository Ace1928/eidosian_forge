import pandas as pd
import sqlite3
import math
import numpy as np
from typing import List, Dict, Any
import yfinance as yf

# Constants
# The path to the database file
DB_PATH = "stock_data.db"


class StockDatabase:
    def __init__(self, db_path: str):
        """Initialize the database connection."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.initialize_database()

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a single SQL query."""
        self.cursor.execute(query, params)
        self.conn.commit()

    def initialize_database(self):
        """Create tables if they do not exist."""
        tables = {
            "Market_Data": """
                CREATE TABLE IF NOT EXISTS Market_Data (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )
            """,
            "Adjusted_Market_Data": """
                CREATE TABLE IF NOT EXISTS Adjusted_Market_Data (
                    ticker TEXT,
                    adjusted_close REAL,
                    adjusted_volume INTEGER
                )
            """,
            "Dividends_Splits": """
                CREATE TABLE IF NOT EXISTS Dividends_Splits (
                    ticker TEXT,
                    dividends REAL,
                    stock_splits REAL
                )
            """,
            "Moving_Averages": """
                CREATE TABLE IF NOT EXISTS Moving_Averages (
                    ticker TEXT,
                    ma_5 REAL,  -- Moving Average 5
                    ma_15 REAL,  -- Moving Average 15
                    ma_50 REAL,  -- Moving Average 50
                    ma_200 REAL,  -- Moving Average 200
                    ma_365 REAL,  -- Moving Average 365
                    ema_5 REAL,  -- Exponential Moving Average 5
                    ema_15 REAL,  -- Exponential Moving Average 15
                    ema_50 REAL,  -- Exponential Moving Average 50
                    ema_200 REAL,  -- Exponential Moving Average 200
                    ema_365 REAL,  -- Exponential Moving Average 365
                    mma_5 REAL,  -- Multiface Moving Average 5
                    mma_15 REAL,  -- Multiface Moving Average 15
                    mma_50 REAL,  -- Multiface Moving Average 50
                    mma_200 REAL,  -- Multiface Moving Average 200
                    mma_365 REAL,  -- Multiface Moving Average 365
                    tema_5 REAL,  -- Triple Exponential Moving Average 5
                    tema_15 REAL,  -- Triple Exponential Moving Average 15
                    tema_50 REAL,  -- Triple Exponential Moving Average 50
                    tema_200 REAL,  -- Triple Exponential Moving Average 200
                    tema_365 REAL  -- Triple Exponential Moving Average 365
                )
            """,
            "Oscillators_Momentum": """
                CREATE TABLE IF NOT EXISTS Oscillators_Momentum (
                    ticker TEXT,
                    rsi REAL,  -- Relative Strength Index (RSI)
                    mfi REAL,  -- Money Flow Index (MFI)
                    tsi REAL,  -- True Strength Index (TSI)
                    uo REAL,  -- Ultimate Oscillator (UO)
                    stoch_rsi REAL,  -- Stochastic RSI
                    stoch_rsi_k REAL,  -- Stochastic RSI %K
                    stoch_rsi_d REAL,  -- Stochastic RSI %D
                    wr REAL,  -- Williams %R
                    ao REAL,  -- Awesome Oscillator
                    kama REAL,  -- Kaufman's Adaptive Moving Average (KAMA)
                    ppo REAL,  -- Percentage Price Oscillator (PPO)
                    ppo_signal REAL,  -- PPO Signal Line
                    ppo_hist REAL,  -- PPO Histogram
                    pvo REAL,  -- Percentage Volume Oscillator (PVO)
                    pvo_signal REAL,  -- PVO Signal Line
                    pvo_hist REAL,  -- PVO Histogram
                    roc REAL,  -- Rate of Change (ROC)
                    roc_100 REAL,  -- Rate of Change (ROC) 100
                    roc_100_sma REAL  -- Rate of Change (ROC) 100 Simple Moving Average (SMA)
                )
            """,
            "Volatility_Indicators": """
                CREATE TABLE IF NOT EXISTS Volatility_Indicators (
                    ticker TEXT,
                    bollinger_bands REAL,  -- Bollinger Bands
                    bollinger_bands_upper REAL,  -- Bollinger Bands Upper Band
                    bollinger_bands_lower REAL,  -- Bollinger Bands Lower Band
                    bollinger_bands_middle REAL,  -- Bollinger Bands Middle Band
                    keltner_channels REAL,  -- Keltner Channels
                    keltner_channels_upper REAL,  -- Keltner Channels Upper Band
                    keltner_channels_lower REAL,  -- Keltner Channels Lower Band
                    keltner_channels_middle REAL,  -- Keltner Channels Middle Band
                    donchian_channels REAL,  -- Donchian Channels
                    donchian_channels_upper REAL,  -- Donchian Channels Upper Band
                    donchian_channels_lower REAL,  -- Donchian Channels Lower Band
                    donchian_channels_middle REAL,  -- Donchian Channels Middle Band
                    atr REAL,  -- ATR(Average True Range)
                    true_range REAL,  -- True Range
                    natr REAL  -- NATR(Normalized Average True Range)
                )
            """,
            "Volume_Indicators": """
                CREATE TABLE IF NOT EXISTS Volume_Indicators (
                    ticker TEXT,
                    adi REAL,  -- ADI(Accumulation Distribution Index)
                    obv REAL,  -- OBV(On Balance Volume)
                    cmf REAL,  -- CMF(Chaikin Money Flow)
                    fi REAL,  -- FI(Force Index)
                    em REAL,  -- EM(Ease of Movement)
                    sma_em REAL,  -- SMA(Simple Moving Average) Ease of Movement
                    vpt REAL,  -- VPT(Volume Price Trend)
                    nvi REAL,  -- NVI(Negative Volume Index)
                    vwap REAL  -- VWAP(Volume Weighted Average Price)
                )
            """,
            "Trend_Indicators": """
                CREATE TABLE IF NOT EXISTS Trend_Indicators (
                    ticker TEXT,
                    parabolic_sar REAL, -- Parabolic SAR
                    directional_movement_index REAL, -- Directional Movement Index
                    minus_directional_indicator REAL, -- Minus Directional Indicator
                    plus_directional_indicator REAL, -- Plus Directional Indicator
                    average_directional_index REAL, -- Average Directional Index
                    adx REAL,  -- ADX
                    adx_pos_di REAL,  -- ADX Positive DI
                    adx_neg_di REAL,  -- ADX Negative DI
                    cci REAL,  -- CCI
                    macd REAL,  -- MACD
                    macd_signal REAL,  -- MACD Signal Line
                    macd_diff REAL,  -- MACD Difference
                    ema_fast REAL,  -- EMA Fast
                    ema_slow REAL,  -- EMA Slow
                    ichimoku_a REAL,  -- Ichimoku A
                    ichimoku_b REAL,  -- Ichimoku B
                    ichimoku_base_line REAL,  -- Ichimoku Base Line
                    ichimoku_conversion_line REAL,  -- Ichimoku Conversion Line
                    kst REAL,  -- KST
                    kst_sig REAL,  -- KST Signal Line
                    kst_diff REAL,  -- KST Difference
                    psar REAL,  -- PSAR
                    psar_up_indicator REAL,  -- PSAR Up Indicator
                    psar_down_indicator REAL,  -- PSAR Down Indicator
                    stc REAL,  -- STC
                    trix REAL,  -- Trix
                    vortex_ind_pos REAL,  -- Vortex Indicator Positive DI
                    vortex_ind_neg REAL,  -- Vortex Indicator Negative DI
                    vortex_ind_diff REAL  -- Vortex Indicator Difference
                )
            """,
            "Price_Patterns_Candlesticks": """
                CREATE TABLE IF NOT EXISTS Price_Patterns_Candlesticks (
                    ticker TEXT,
                    cdl_2_crows REAL,  -- CDL 2 Crows
                    cdl_3_black_crows REAL,  -- CDL 3 Black Crows
                    cdl_3_inside REAL,  -- CDL 3 Inside
                    cdl_3_line_strike REAL,  -- CDL 3 Line Strike
                    cdl_3_outside REAL,  -- CDL 3 Outside
                    cdl_3_stars_in_south REAL,  -- CDL 3 Stars In South
                    cdl_3_white_soldiers REAL,  -- CDL 3 White Soldiers
                    cdl_abandoned_baby REAL,  -- CDL Abandoned Baby
                    cdl_advance_block REAL,  -- CDL Advance Block
                    cdl_belt_hold REAL,  -- CDL Belt Hold
                    cdl_breakaway REAL,  -- CDL Breakaway
                    cdl_closing_marubozu REAL,  -- CDL Closing Marubozu
                    cdl_conceal_baby_swall REAL,  -- CDL Conceal Baby Swall
                    cdl_counterattack REAL,  -- CDL Counterattack
                    cdl_dark_cloud_cover REAL,  -- CDL Dark Cloud Cover
                    cdl_doji REAL,  -- CDL Doji
                    cdl_doji_star REAL,  -- CDL Doji Star
                    cdl_dragonfly_doji REAL,  -- CDL Dragonfly Doji
                    cdl_engulfing REAL,  -- CDL Engulfing
                    cdl_evening_doji_star REAL,  -- CDL Evening Doji Star
                    cdl_evening_star REAL,  -- CDL Evening Star
                    cdl_gap_side_side_white REAL,  -- CDL Gap Side Side White
                    cdl_gravestone_doji REAL,  -- CDL Gravestone Doji
                    cdl_hammer REAL,  -- CDL Hammer
                    cdl_hanging_man REAL,  -- CDL Hanging Man
                    cdl_harami REAL,  -- CDL Harami
                    cdl_harami_cross REAL,  -- CDL Harami Cross
                    cdl_high_wave REAL,  -- CDL High Wave
                    cdl_hikkake REAL,  -- CDL Hikkake
                    cdl_hikkake_modified REAL,  -- CDL Hikkake Modified
                    cdl_homing_pigeon REAL,  -- CDL Homing Pigeon
                    cdl_identical_3_crows REAL,  -- CDL Identical 3 Crows
                    cdl_in_neck REAL,  -- CDL In Neck
                    cdl_inverted_hammer REAL,  -- CDL Inverted Hammer
                    cdl_kicking REAL,  -- CDL Kicking
                    cdl_kicking_by_length REAL,  -- CDL Kicking By Length
                    cdl_ladder_bottom REAL,  -- CDL Ladder Bottom
                    cdl_long_legged_doji REAL,  -- CDL Long Legged Doji
                    cdl_long_line REAL,  -- CDL Long Line
                    cdl_marubozu REAL,  -- CDL Marubozu
                    cdl_matching_low REAL,  -- CDL Matching Low
                    cdl_mat_hold REAL,  -- CDL Mat Hold
                    cdl_morning_doji_star REAL,  -- CDL Morning Doji Star
                    cdl_morning_star REAL,  -- CDL Morning Star
                    cdl_on_neck REAL,  -- CDL On Neck
                    cdl_piercing REAL,  -- CDL Piercing
                    cdl_rickshaw_man REAL,  -- CDL Rickshaw Man
                    cdl_rise_fall_3_methods REAL,  -- CDL Rise Fall 3 Methods
                    cdl_separating_lines REAL,  -- CDL Separating Lines
                    cdl_shooting_star REAL,  -- CDL Shooting Star
                    cdl_short_line REAL,  -- CDL Short Line
                    cdl_spinning_top REAL,  -- CDL Spinning Top
                    cdl_stalled_pattern REAL,  -- CDL Stalled Pattern
                    cdl_stick_sandwich REAL,  -- CDL Stick Sandwich
                    cdl_takuri REAL,  -- CDL Takuri
                    cdl_tasuki_gap REAL,  -- CDL Tasuki Gap
                    cdl_thrusting REAL,  -- CDL Thrusting
                    cdl_tristar REAL,  -- CDL Tristar
                    cdl_unique_3_river REAL,  -- CDL Unique 3 River
                    cdl_upside_gap_2_crows REAL,  -- CDL Upside Gap 2 Crows
                    cdl_x_side_gap_3_methods REAL  -- CDL X Side Gap 3 Methods
                )
            """,
            "Advanced_Statistical_Measures": """
                CREATE TABLE IF NOT EXISTS Advanced_Statistical_Measures (
                    ticker TEXT,
                    beta REAL,  -- Beta
                    correlation_coefficient REAL,  -- Correlation Coefficient
                    linear_regression_angle REAL,  -- Linear Regression Angle
                    linear_regression_intercept REAL,  -- Linear Regression Intercept
                    linear_regression_slope REAL,  -- Linear Regression Slope
                    standard_deviation REAL,  -- Standard Deviation
                    standard_error REAL,  -- Standard Error
                    time_series_forecast REAL,  -- Time Series Forecast
                    variance REAL  -- Variance
                )
            """,
            "Math_Transformations": """
                CREATE TABLE IF NOT EXISTS Math_Transformations (
                    ticker TEXT,
                    transform_acos REAL,  -- Transform ACOS
                    transform_asin REAL,  -- Transform ASIN
                    transform_atan REAL,  -- Transform ATAN
                    transform_ceil REAL,  -- Transform CEIL
                    transform_cos REAL,  -- Transform COS
                    transform_cosh REAL,  -- Transform COSH
                    transform_exp REAL,  -- Transform EXP
                    transform_floor REAL,  -- Transform FLOOR
                    transform_ln REAL,  -- Transform LN
                    transform_log10 REAL,  -- Transform LOG10
                    transform_sin REAL,  -- Transform SIN
                    transform_sinh REAL,  -- Transform SINH
                    transform_sqrt REAL,  -- Transform SQRT
                    transform_tan REAL  -- Transform TAN
                )
            """,
            "Corporate_Events": """
                CREATE TABLE IF NOT EXISTS Corporate_Events (
                    ticker TEXT,
                    event_type TEXT,  -- Event Type (e.g., earnings release, product launch)
                    event_date TEXT,  -- Date of the Event
                    impact_score REAL  -- Quantitative Measure of Expected Impact
                )
            """,
            "Market_Sentiment": """
                CREATE TABLE IF NOT EXISTS Market_Sentiment (
                    ticker TEXT,
                    sentiment_score REAL,  -- Sentiment Score
                    sentiment_volume INTEGER,  -- Sentiment Volume
                    date TEXT  -- Date of the Sentiment
                )
            """,
            "Risk_Metrics": """
                CREATE TABLE IF NOT EXISTS Risk_Metrics (
                    ticker TEXT,
                    beta REAL,  -- Beta
                    alpha REAL,  -- Alpha
                    sharpe_ratio REAL,  -- Sharpe Ratio
                    sortino_ratio REAL,  -- Sortino Ratio
                    date TEXT  -- Date of the Risk Metrics
                )
            """,
            "Trading_Sessions": """
                CREATE TABLE IF NOT EXISTS Trading_Sessions (
                    ticker TEXT,
                    session_date TEXT,  -- Date of the Trading Session
                    open_price REAL,  -- Opening Price
                    close_price REAL,  -- Closing Price
                    session_high REAL,  -- Highest Price During the Session
                    session_low REAL,  -- Lowest Price During the Session
                    transaction_volume INTEGER  -- Volume of Transactions During the Session
                )
            """,
            "Derivatives_Options": """
                CREATE TABLE IF NOT EXISTS Derivatives_Options (
                    ticker TEXT,
                    option_type TEXT,  -- Call or Put
                    strike_price REAL,  -- Price at Which the Option Can be Exercised
                    expiration_date TEXT,  -- Date When the Option Expires
                    open_interest INTEGER,  -- Number of Open Contracts
                    implied_volatility REAL  -- Measure of Expected Volatility
                )
            """,
            "Sector_Industry": """
                CREATE TABLE IF NOT EXISTS Sector_Industry (
                    ticker TEXT,
                    sector TEXT,  -- Sector of the Stock
                    industry TEXT,  -- Industry of the Stock
                    market_cap REAL,  -- Market Capitalization
                    employee_count INTEGER  -- Number of Employees
                )
            """,
            "Financial_Ratios": """
                CREATE TABLE IF NOT EXISTS Financial_Ratios (
                    ticker TEXT,
                    price_to_earnings REAL,  -- Price-to-Earnings Ratio
                    return_on_equity REAL,  -- Return on Equity
                    debt_to_equity REAL,  -- Debt-to-Equity Ratio
                    current_ratio REAL,  -- Current Ratio
                    date TEXT  -- Date of the Financial Ratios
                )
            """,
            # Each table creation string follows the pattern established.
        }
        for table, query in tables.items():
            self.execute_query(query)

    def update_table(self, table: str, data: Dict[str, Any]):
        """Update a specific table with a dictionary of data."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute_query(query, tuple(data.values()))

    def fetch_data(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch data from the database."""
        self.cursor.execute(query, params)
        columns = [column[0] for column in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]


# Example usage
db = StockDatabase(DB_PATH)
import yfinance as yf
import pandas as pd
import sqlite3

# Your database class and methods would go here

# Downloading the data
data = yf.download(tickers="AAPL", start="2022-01-01", end="2022-12-31", interval="1d")
data = data.reset_index()
data["ticker"] = "AAPL"
data = data.rename(
    columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
)
# Convert the date column to string
data["date"] = data["date"].dt.strftime("%Y-%m-%d")

# Prepare data for insertion
data = data[["ticker", "date", "open", "high", "low", "close", "volume"]]
data = data.to_dict(orient="records")

# Assuming db is an instance of StockDatabase
for row in data:
    db.update_table("Market_Data", row)

print(db.fetch_data("SELECT * FROM Market_Data WHERE ticker = 'AAPL'"))

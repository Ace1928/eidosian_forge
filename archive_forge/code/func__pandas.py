import pandas as pd
import numpy as np
def _pandas(mode, trendline_options, x_raw, y, non_missing):
    modes = dict(rolling='Rolling', ewm='Exponentially Weighted', expanding='Expanding')
    trendline_options = trendline_options.copy()
    function_name = trendline_options.pop('function', 'mean')
    function_args = trendline_options.pop('function_args', dict())
    series = pd.Series(y, index=x_raw)
    agg = getattr(series, mode)
    agg_obj = agg(**trendline_options)
    function = getattr(agg_obj, function_name)
    y_out = function(**function_args)
    y_out = y_out[non_missing]
    hover_header = '<b>%s %s trendline</b><br><br>' % (modes[mode], function_name)
    return (y_out, hover_header, None)
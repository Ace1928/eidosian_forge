from plotly.figure_factory import utils
from plotly.figure_factory._ohlc import (
from plotly.graph_objs import graph_objs
class _Candlestick(object):
    """
    Refer to FigureFactory.create_candlestick() for docstring.
    """

    def __init__(self, open, high, low, close, dates, **kwargs):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        if dates is not None:
            self.x = dates
        else:
            self.x = [x for x in range(len(self.open))]
        self.get_candle_increase()

    def get_candle_increase(self):
        """
        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        """
        increase_y = []
        increase_x = []
        for index in range(len(self.open)):
            if self.close[index] > self.open[index]:
                increase_y.append(self.low[index])
                increase_y.append(self.open[index])
                increase_y.append(self.close[index])
                increase_y.append(self.close[index])
                increase_y.append(self.close[index])
                increase_y.append(self.high[index])
                increase_x.append(self.x[index])
        increase_x = [[x, x, x, x, x, x] for x in increase_x]
        increase_x = utils.flatten(increase_x)
        return (increase_x, increase_y)

    def get_candle_decrease(self):
        """
        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        """
        decrease_y = []
        decrease_x = []
        for index in range(len(self.open)):
            if self.close[index] <= self.open[index]:
                decrease_y.append(self.low[index])
                decrease_y.append(self.open[index])
                decrease_y.append(self.close[index])
                decrease_y.append(self.close[index])
                decrease_y.append(self.close[index])
                decrease_y.append(self.high[index])
                decrease_x.append(self.x[index])
        decrease_x = [[x, x, x, x, x, x] for x in decrease_x]
        decrease_x = utils.flatten(decrease_x)
        return (decrease_x, decrease_y)
from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
class _OHLC(object):
    """
    Refer to FigureFactory.create_ohlc_increase() for docstring.
    """

    def __init__(self, open, high, low, close, dates, **kwargs):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.empty = [None] * len(open)
        self.dates = dates
        self.all_x = []
        self.all_y = []
        self.increase_x = []
        self.increase_y = []
        self.decrease_x = []
        self.decrease_y = []
        self.get_all_xy()
        self.separate_increase_decrease()

    def get_all_xy(self):
        """
        Zip data to create OHLC shape

        OHLC shape: low to high vertical bar with
        horizontal branches for open and close values.
        If dates were added, the smallest date difference is calculated and
        multiplied by .2 to get the length of the open and close branches.
        If no date data was provided, the x-axis is a list of integers and the
        length of the open and close branches is .2.
        """
        self.all_y = list(zip(self.open, self.open, self.high, self.low, self.close, self.close, self.empty))
        if self.dates is not None:
            date_dif = []
            for i in range(len(self.dates) - 1):
                date_dif.append(self.dates[i + 1] - self.dates[i])
            date_dif_min = min(date_dif) / 5
            self.all_x = [[x - date_dif_min, x, x, x, x, x + date_dif_min, None] for x in self.dates]
        else:
            self.all_x = [[x - 0.2, x, x, x, x, x + 0.2, None] for x in range(len(self.open))]

    def separate_increase_decrease(self):
        """
        Separate data into two groups: increase and decrease

        (1) Increase, where close > open and
        (2) Decrease, where close <= open
        """
        for index in range(len(self.open)):
            if self.close[index] is None:
                pass
            elif self.close[index] > self.open[index]:
                self.increase_x.append(self.all_x[index])
                self.increase_y.append(self.all_y[index])
            else:
                self.decrease_x.append(self.all_x[index])
                self.decrease_y.append(self.all_y[index])

    def get_increase(self):
        """
        Flatten increase data and get increase text

        :rtype (list, list, list): flat_increase_x: x-values for the increasing
            trace, flat_increase_y: y=values for the increasing trace and
            text_increase: hovertext for the increasing trace
        """
        flat_increase_x = utils.flatten(self.increase_x)
        flat_increase_y = utils.flatten(self.increase_y)
        text_increase = ('Open', 'Open', 'High', 'Low', 'Close', 'Close', '') * len(self.increase_x)
        return (flat_increase_x, flat_increase_y, text_increase)

    def get_decrease(self):
        """
        Flatten decrease data and get decrease text

        :rtype (list, list, list): flat_decrease_x: x-values for the decreasing
            trace, flat_decrease_y: y=values for the decreasing trace and
            text_decrease: hovertext for the decreasing trace
        """
        flat_decrease_x = utils.flatten(self.decrease_x)
        flat_decrease_y = utils.flatten(self.decrease_y)
        text_decrease = ('Open', 'Open', 'High', 'Low', 'Close', 'Close', '') * len(self.decrease_x)
        return (flat_decrease_x, flat_decrease_y, text_decrease)
from plotly.utils import _list_repr_elided
class BoxSelector:

    def __init__(self, xrange=None, yrange=None, **_):
        self._type = 'box'
        self._xrange = xrange
        self._yrange = yrange

    def __repr__(self):
        return 'BoxSelector(xrange={xrange},\n            yrange={yrange})'.format(xrange=self.xrange, yrange=self.yrange)

    @property
    def type(self):
        """
        The selector's type

        Returns
        -------
        str
        """
        return self._type

    @property
    def xrange(self):
        """
        x-axis range extents of the box selection

        Returns
        -------
        (float, float)
        """
        return self._xrange

    @property
    def yrange(self):
        """
        y-axis range extents of the box selection

        Returns
        -------
        (float, float)
        """
        return self._yrange
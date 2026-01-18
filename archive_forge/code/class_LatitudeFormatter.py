import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
class LatitudeFormatter(_PlateCarreeFormatter):
    """Tick formatter for latitude axes."""

    def __init__(self, direction_label=True, degree_symbol='°', number_format='g', transform_precision=1e-08, dms=False, minute_symbol='′', second_symbol='″', seconds_number_format='g', auto_hide=True, decimal_point=None, cardinal_labels=None):
        """
        Tick formatter for latitudes.

        When bound to an axis, the axis must be part of an axes defined
        on a rectangular projection (e.g. Plate Carree, Mercator).


        Parameters
        ----------
        direction_label: optional
            If *True* a direction label (N or S) will be drawn next to
            latitude labels. If *False* then these
            labels will not be drawn. Defaults to *True* (draw direction
            labels).
        degree_symbol: optional
            The character(s) used to represent the degree symbol in the tick
            labels. Defaults to '°'. Can be an empty string if no degree symbol
            is desired.
        number_format: optional
            Format string to represent the longitude values when `dms`
            is set to False. Defaults to 'g'.
        transform_precision: optional
            Sets the precision (in degrees) to which transformed tick
            values are rounded. The default is 1e-7, and should be
            suitable for most use cases. To control the appearance of
            tick labels use the *number_format* keyword.
        dms: bool, optional
            Whether or not formatting as degrees-minutes-seconds and not
            as decimal degrees.
        minute_symbol: str, optional
            The character(s) used to represent the minute symbol.
        second_symbol: str, optional
            The character(s) used to represent the second symbol.
        seconds_number_format: optional
            Format string to represent the "seconds" component of the longitude
            values. Defaults to 'g'.
        auto_hide: bool, optional
            Auto-hide degrees or minutes when redundant.
        decimal_point: bool, optional
            Decimal point character. If not provided and
            ``mpl.rcParams['axes.formatter.use_locale'] == True``,
            the locale decimal point is used.
        cardinal_labels: dict, optional
            A dictionary with "south" and/or "north" keys to replace south and
            north cardinal labels, which defaults to "S" and "N".

        Note
        ----
            A formatter can only be used for one axis. A new formatter
            must be created for every axis that needs formatted labels.

        Examples
        --------
        Label latitudes from -90 to 90 on a Plate Carree projection::

            ax = plt.axes(projection=PlateCarree())
            ax.set_global()
            ax.set_yticks([-90, -60, -30, 0, 30, 60, 90],
                          crs=ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)

        Label latitudes from -80 to 80 on a Mercator projection, this
        time omitting the degree symbol::

            ax = plt.axes(projection=Mercator())
            ax.set_global()
            ax.set_yticks([-90, -60, -30, 0, 30, 60, 90],
                          crs=ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter(degree_symbol='')
            ax.yaxis.set_major_formatter(lat_formatter)

        When not bound to an axis::

            lat_formatter = LatitudeFormatter()
            ticks = [-90, -60, -30, 0, 30, 60, 90]
            lat_formatter.set_locs(ticks)
            labels = [lat_formatter(value) for value in ticks]

        """
        super().__init__(direction_label=direction_label, degree_symbol=degree_symbol, number_format=number_format, transform_precision=transform_precision, dms=dms, minute_symbol=minute_symbol, second_symbol=second_symbol, seconds_number_format=seconds_number_format, auto_hide=auto_hide, decimal_point=decimal_point, cardinal_labels=cardinal_labels)

    def _apply_transform(self, value, target_proj, source_crs):
        return target_proj.transform_point(0, value, source_crs)[1]

    def _hemisphere(self, value, value_source_crs):
        if value > 0:
            hemisphere = self._cardinal_labels.get('north', 'N')
        elif value < 0:
            hemisphere = self._cardinal_labels.get('south', 'S')
        else:
            hemisphere = ''
        return hemisphere
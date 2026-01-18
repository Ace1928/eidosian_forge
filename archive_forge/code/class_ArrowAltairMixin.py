from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence, cast
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import (
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
class ArrowAltairMixin:

    @gather_metrics('line_chart')
    def line_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | list[Color] | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        """Display a line chart.

        This is syntax-sugar around ``st.altair_chart``. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If ``st.line_chart`` does not guess the data specification
        correctly, try specifying your desired chart using ``st.altair_chart``.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, dict or None
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.

        y : str, Sequence of str, or None
            Column name(s) to use for the y-axis. If a Sequence of strings,
            draws several series on the same chart by melting your wide-format
            table into a long-format table behind the scenes. If None, draws
            the data of all remaining columns as data series.

        color : str, tuple, Sequence of str, Sequence of tuple, or None
            The color to use for different lines in this chart.

            For a line chart with just one line, this can be:

            * None, to use the default color.
            * A hex string like "#ffaa00" or "#ffaa0088".
            * An RGB or RGBA tuple with the red, green, blue, and alpha
              components specified as ints from 0 to 255 or floats from 0.0 to
              1.0.

            For a line chart with multiple lines, where the dataframe is in
            long format (that is, y is None or just one column), this can be:

            * None, to use the default colors.
            * The name of a column in the dataset. Data points will be grouped
              into lines of the same color based on the value of this column.
              In addition, if the values in this column match one of the color
              formats above (hex string or color tuple), then that color will
              be used.

              For example: if the dataset has 1000 rows, but this column only
              contains the values "adult", "child", and "baby", then those 1000
              datapoints will be grouped into three lines whose colors will be
              automatically selected from the default palette.

              But, if for the same 1000-row dataset, this column contained
              the values "#ffaa00", "#f0f", "#0000ff", then then those 1000
              datapoints would still be grouped into three lines, but their
              colors would be "#ffaa00", "#f0f", "#0000ff" this time around.

            For a line chart with multiple lines, where the dataframe is in
            wide format (that is, y is a Sequence of columns), this can be:

            * None, to use the default colors.
            * A list of string colors or color tuples to be used for each of
              the lines in the chart. This list should have the same length
              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``
              for three lines).

        width : int
            The chart width in pixels. If 0, selects the width automatically.

        height : int
            The chart height in pixels. If 0, selects the height automatically.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        >>>
        >>> st.line_chart(chart_data)

        .. output::
           https://doc-line-chart.streamlit.app/
           height: 440px

        You can also choose different columns to use for x and y, as well as set
        the color dynamically based on a 3rd column (assuming your dataframe is in
        long format):

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...    {
        ...        "col1": np.random.randn(20),
        ...        "col2": np.random.randn(20),
        ...        "col3": np.random.choice(["A", "B", "C"], 20),
        ...    }
        ... )
        >>>
        >>> st.line_chart(chart_data, x="col1", y="col2", color="col3")

        .. output::
           https://doc-line-chart1.streamlit.app/
           height: 440px

        Finally, if your dataframe is in wide format, you can group multiple
        columns under the y argument to show multiple lines with different
        colors:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])
        >>>
        >>> st.line_chart(
        ...    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
        ... )

        .. output::
           https://doc-line-chart2.streamlit.app/
           height: 440px

        """
        proto = ArrowVegaLiteChartProto()
        chart, add_rows_metadata = _generate_chart(chart_type=ChartType.LINE, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=None, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_line_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('area_chart')
    def area_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | list[Color] | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        """Display an area chart.

        This is syntax-sugar around ``st.altair_chart``. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If ``st.area_chart`` does not guess the data specification
        correctly, try specifying your desired chart using ``st.altair_chart``.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, or dict
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.

        y : str, Sequence of str, or None
            Column name(s) to use for the y-axis. If a Sequence of strings,
            draws several series on the same chart by melting your wide-format
            table into a long-format table behind the scenes. If None, draws
            the data of all remaining columns as data series.

        color : str, tuple, Sequence of str, Sequence of tuple, or None
            The color to use for different series in this chart.

            For an area chart with just 1 series, this can be:

            * None, to use the default color.
            * A hex string like "#ffaa00" or "#ffaa0088".
            * An RGB or RGBA tuple with the red, green, blue, and alpha
              components specified as ints from 0 to 255 or floats from 0.0 to
              1.0.

            For an area chart with multiple series, where the dataframe is in
            long format (that is, y is None or just one column), this can be:

            * None, to use the default colors.
            * The name of a column in the dataset. Data points will be grouped
              into series of the same color based on the value of this column.
              In addition, if the values in this column match one of the color
              formats above (hex string or color tuple), then that color will
              be used.

              For example: if the dataset has 1000 rows, but this column only
              contains the values "adult", "child", and "baby", then those 1000
              datapoints will be grouped into three series whose colors will be
              automatically selected from the default palette.

              But, if for the same 1000-row dataset, this column contained
              the values "#ffaa00", "#f0f", "#0000ff", then then those 1000
              datapoints would still be grouped into 3 series, but their
              colors would be "#ffaa00", "#f0f", "#0000ff" this time around.

            For an area chart with multiple series, where the dataframe is in
            wide format (that is, y is a Sequence of columns), this can be:

            * None, to use the default colors.
            * A list of string colors or color tuples to be used for each of
              the series in the chart. This list should have the same length
              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``
              for three lines).

        width : int
            The chart width in pixels. If 0, selects the width automatically.

        height : int
            The chart height in pixels. If 0, selects the height automatically.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        >>>
        >>> st.area_chart(chart_data)

        .. output::
           https://doc-area-chart.streamlit.app/
           height: 440px

        You can also choose different columns to use for x and y, as well as set
        the color dynamically based on a 3rd column (assuming your dataframe is in
        long format):

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...    {
        ...        "col1": np.random.randn(20),
        ...        "col2": np.random.randn(20),
        ...        "col3": np.random.choice(["A", "B", "C"], 20),
        ...    }
        ... )
        >>>
        >>> st.area_chart(chart_data, x="col1", y="col2", color="col3")

        .. output::
           https://doc-area-chart1.streamlit.app/
           height: 440px

        Finally, if your dataframe is in wide format, you can group multiple
        columns under the y argument to show multiple series with different
        colors:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])
        >>>
        >>> st.area_chart(
        ...    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
        ... )

        .. output::
           https://doc-area-chart2.streamlit.app/
           height: 440px

        """
        proto = ArrowVegaLiteChartProto()
        chart, add_rows_metadata = _generate_chart(chart_type=ChartType.AREA, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=None, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_area_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('bar_chart')
    def bar_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | list[Color] | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        """Display a bar chart.

        This is syntax-sugar around ``st.altair_chart``. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If ``st.bar_chart`` does not guess the data specification
        correctly, try specifying your desired chart using ``st.altair_chart``.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, or dict
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.

        y : str, Sequence of str, or None
            Column name(s) to use for the y-axis. If a Sequence of strings,
            draws several series on the same chart by melting your wide-format
            table into a long-format table behind the scenes. If None, draws
            the data of all remaining columns as data series.

        color : str, tuple, Sequence of str, Sequence of tuple, or None
            The color to use for different series in this chart.

            For a bar chart with just one series, this can be:

            * None, to use the default color.
            * A hex string like "#ffaa00" or "#ffaa0088".
            * An RGB or RGBA tuple with the red, green, blue, and alpha
              components specified as ints from 0 to 255 or floats from 0.0 to
              1.0.

            For a bar chart with multiple series, where the dataframe is in
            long format (that is, y is None or just one column), this can be:

            * None, to use the default colors.
            * The name of a column in the dataset. Data points will be grouped
              into series of the same color based on the value of this column.
              In addition, if the values in this column match one of the color
              formats above (hex string or color tuple), then that color will
              be used.

              For example: if the dataset has 1000 rows, but this column only
              contains the values "adult", "child", and "baby", then those 1000
              datapoints will be grouped into three series whose colors will be
              automatically selected from the default palette.

              But, if for the same 1000-row dataset, this column contained
              the values "#ffaa00", "#f0f", "#0000ff", then then those 1000
              datapoints would still be grouped into 3 series, but their
              colors would be "#ffaa00", "#f0f", "#0000ff" this time around.

            For a bar chart with multiple series, where the dataframe is in
            wide format (that is, y is a Sequence of columns), this can be:

            * None, to use the default colors.
            * A list of string colors or color tuples to be used for each of
              the series in the chart. This list should have the same length
              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``
              for three lines).

        width : int
            The chart width in pixels. If 0, selects the width automatically.

        height : int
            The chart height in pixels. If 0, selects the height automatically.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        >>>
        >>> st.bar_chart(chart_data)

        .. output::
           https://doc-bar-chart.streamlit.app/
           height: 440px

        You can also choose different columns to use for x and y, as well as set
        the color dynamically based on a 3rd column (assuming your dataframe is in
        long format):

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...    {
        ...        "col1": list(range(20)) * 3,
        ...        "col2": np.random.randn(60),
        ...        "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        ...    }
        ... )
        >>>
        >>> st.bar_chart(chart_data, x="col1", y="col2", color="col3")

        .. output::
           https://doc-bar-chart1.streamlit.app/
           height: 440px

        Finally, if your dataframe is in wide format, you can group multiple
        columns under the y argument to show multiple series with different
        colors:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...    {"col1": list(range(20)), "col2": np.random.randn(20), "col3": np.random.randn(20)}
        ... )
        >>>
        >>> st.bar_chart(
        ...    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
        ... )

        .. output::
           https://doc-bar-chart2.streamlit.app/
           height: 440px

        """
        proto = ArrowVegaLiteChartProto()
        chart, add_rows_metadata = _generate_chart(chart_type=ChartType.BAR, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=None, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_bar_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('scatter_chart')
    def scatter_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | list[Color] | None=None, size: str | float | int | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        """Display a scatterplot chart.

        This is syntax-sugar around ``st.altair_chart``. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If ``st.scatter_chart`` does not guess the data specification correctly,
        try specifying your desired chart using ``st.altair_chart``.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, dict or None
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.

        y : str, Sequence of str, or None
            Column name(s) to use for the y-axis. If a Sequence of strings,
            draws several series on the same chart by melting your wide-format
            table into a long-format table behind the scenes. If None, draws
            the data of all remaining columns as data series.

        color : str, tuple, Sequence of str, Sequence of tuple, or None
            The color of the circles representing each datapoint.

            This can be:

            * None, to use the default color.
            * A hex string like "#ffaa00" or "#ffaa0088".
            * An RGB or RGBA tuple with the red, green, blue, and alpha
              components specified as ints from 0 to 255 or floats from 0.0 to
              1.0.
            * The name of a column in the dataset where the color of that
              datapoint will come from.

              If the values in this column are in one of the color formats
              above (hex string or color tuple), then that color will be used.

              Otherwise, the color will be automatically picked from the
              default palette.

              For example: if the dataset has 1000 rows, but this column only
              contains the values "adult", "child", and "baby", then those 1000
              datapoints be shown using three colors from the default palette.

              But if this column only contains floats or ints, then those
              1000 datapoints will be shown using a colors from a continuous
              color gradient.

              Finally, if this column only contains the values "#ffaa00",
              "#f0f", "#0000ff", then then each of those 1000 datapoints will
              be assigned "#ffaa00", "#f0f", or "#0000ff" as appropriate.

            If the dataframe is in wide format (that is, y is a Sequence of
            columns), this can also be:

            * A list of string colors or color tuples to be used for each of
              the series in the chart. This list should have the same length
              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``
              for three series).

        size : str, float, int, or None
            The size of the circles representing each point.

            This can be:

            * A number like 100, to specify a single size to use for all
              datapoints.
            * The name of the column to use for the size. This allows each
              datapoint to be represented by a circle of a different size.

        width : int
            The chart width in pixels. If 0, selects the width automatically.

        height : int
            The chart height in pixels. If 0, selects the height automatically.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        >>>
        >>> st.scatter_chart(chart_data)

        .. output::
           https://doc-scatter-chart.streamlit.app/
           height: 440px

        You can also choose different columns to use for x and y, as well as set
        the color dynamically based on a 3rd column (assuming your dataframe is in
        long format):

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])
        >>> chart_data['col4'] = np.random.choice(['A','B','C'], 20)
        >>>
        >>> st.scatter_chart(
        ...     chart_data,
        ...     x='col1',
        ...     y='col2',
        ...     color='col4',
        ...     size='col3',
        ... )

        .. output::
           https://doc-scatter-chart1.streamlit.app/
           height: 440px

        Finally, if your dataframe is in wide format, you can group multiple
        columns under the y argument to show multiple series with different
        colors:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 4), columns=["col1", "col2", "col3", "col4"])
        >>>
        >>> st.scatter_chart(
        ...     chart_data,
        ...     x='col1',
        ...     y=['col2', 'col3'],
        ...     size='col4',
        ...     color=['#FF0000', '#0000FF'],  # Optional
        ... )

        .. output::
           https://doc-scatter-chart2.streamlit.app/
           height: 440px

        """
        proto = ArrowVegaLiteChartProto()
        chart, add_rows_metadata = _generate_chart(chart_type=ChartType.SCATTER, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=size, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_scatter_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('altair_chart')
    def altair_chart(self, altair_chart: alt.Chart, use_container_width: bool=False, theme: Literal['streamlit'] | None='streamlit') -> DeltaGenerator:
        """Display a chart using the Altair library.

        Parameters
        ----------
        altair_chart : altair.Chart
            The Altair chart object to display.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over Altair's native ``width`` value.

        theme : "streamlit" or None
            The theme of the chart. Currently, we only support "streamlit" for the Streamlit
            defined design or None to fallback to the default behavior of the library.

        Example
        -------

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>> import altair as alt
        >>>
        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        >>>
        >>> c = (
        ...    alt.Chart(chart_data)
        ...    .mark_circle()
        ...    .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
        ... )
        >>>
        >>> st.altair_chart(c, use_container_width=True)

        .. output::
           https://doc-vega-lite-chart.streamlit.app/
           height: 300px

        Examples of Altair charts can be found at
        https://altair-viz.github.io/gallery/.

        """
        if theme != 'streamlit' and theme != None:
            raise StreamlitAPIException(f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.')
        proto = ArrowVegaLiteChartProto()
        marshall(proto, altair_chart, use_container_width=use_container_width, theme=theme)
        return self.dg._enqueue('arrow_vega_lite_chart', proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)
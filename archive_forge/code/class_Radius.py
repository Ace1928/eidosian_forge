import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
@with_property_setters
class Radius(FieldChannelMixin, core.PositionFieldDefBase):
    """Radius schema wrapper

    Parameters
    ----------

    shorthand : str, dict, Sequence[str], :class:`RepeatRef`
        shorthand for field, aggregate, and type
    aggregate : dict, :class:`Aggregate`, :class:`ArgmaxDef`, :class:`ArgminDef`, :class:`NonArgAggregateOp`, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    bandPosition : float
        Relative position on a band of a stacked, binned, time unit, or band scale. For
        example, the marks will be positioned at the beginning of the band if set to ``0``,
        and at the middle of the band if set to ``0.5``.
    bin : str, bool, dict, None, :class:`BinParams`
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#bin-parameters>`__, or indicating
        that the data for ``x`` or ``y`` channel are binned before they are imported into
        Vega-Lite ( ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#bin-parameters>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : str, dict, :class:`Field`, :class:`FieldName`, :class:`RepeatRef`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    scale : dict, None, :class:`Scale`
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : dict, None, :class:`Sort`, Sequence[str], Sequence[bool], Sequence[float], :class:`SortArray`, :class:`SortOrder`, :class:`AllSortString`, :class:`SortByChannel`, :class:`SortByEncoding`, :class:`EncodingSortField`, :class:`SortByChannelDesc`, Sequence[dict, :class:`DateTime`], Literal['ascending', 'descending'], Literal['x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity', 'strokeOpacity', 'opacity', 'text'], Literal['-x', '-y', '-color', '-fill', '-stroke', '-strokeWidth', '-size', '-shape', '-fillOpacity', '-strokeOpacity', '-opacity', '-text']
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following:


        * ``"ascending"`` or ``"descending"`` -- for sorting by the values' natural order in
          JavaScript.
        * `A string indicating an encoding channel name to sort by
          <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g.,
          ``"x"`` or ``"y"`` ) with an optional minus prefix for descending sort (e.g.,
          ``"-x"`` to sort by x-field, descending). This channel string is short-form of `a
          sort-by-encoding definition
          <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For
          example, ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
          "descending"}``.
        * `A sort field definition
          <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
          another field.
        * `An array specifying the field values in preferred order
          <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
          sort order will obey the values in the array, followed by any unspecified values
          in their original order. For discrete time field, values in the sort array can be
          `date-time definition objects
          <https://vega.github.io/vega-lite/docs/datetime.html>`__. In addition, for time
          units ``"month"`` and ``"day"``, the values can be the month or day names (case
          insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ).
        * ``null`` indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    stack : bool, None, :class:`StackOffset`, Literal['zero', 'center', 'normalize']
        Type of stacking offset if the field should be stacked. ``stack`` is only applicable
        for ``x``, ``y``, ``theta``, and ``radius`` channels with continuous domains. For
        example, ``stack`` of ``y`` can be used to customize stacking for a vertical bar
        chart.

        ``stack`` can be one of the following values:


        * ``"zero"`` or `true`: stacking with baseline offset at zero value of the scale
          (for creating typical stacked
          [bar](https://vega.github.io/vega-lite/docs/stack.html#bar) and `area
          <https://vega.github.io/vega-lite/docs/stack.html#area>`__ chart).
        * ``"normalize"`` - stacking with normalized domain (for creating `normalized
          stacked bar and area charts
          <https://vega.github.io/vega-lite/docs/stack.html#normalized>`__ and pie charts
          `with percentage tooltip
          <https://vega.github.io/vega-lite/docs/arc.html#tooltip>`__ ). :raw-html:`<br/>`
        * ``"center"`` - stacking with center baseline (for `streamgraph
          <https://vega.github.io/vega-lite/docs/stack.html#streamgraph>`__ ).
        * ``null`` or ``false`` - No-stacking. This will produce layered `bar
          <https://vega.github.io/vega-lite/docs/stack.html#layered-bar-chart>`__ and area
          chart.

        **Default value:** ``zero`` for plots with all of the following conditions are true:
        (1) the mark is ``bar``, ``area``, or ``arc`` ; (2) the stacked measure channel (x
        or y) has a linear scale; (3) At least one of non-position channels mapped to an
        unaggregated field that is different from x and y. Otherwise, ``null`` by default.

        **See also:** `stack <https://vega.github.io/vega-lite/docs/stack.html>`__
        documentation.
    timeUnit : dict, :class:`TimeUnit`, :class:`MultiTimeUnit`, :class:`BinnedTimeUnit`, :class:`SingleTimeUnit`, :class:`TimeUnitParams`, :class:`UtcMultiTimeUnit`, :class:`UtcSingleTimeUnit`, :class:`LocalMultiTimeUnit`, :class:`LocalSingleTimeUnit`, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds']
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : str, None, :class:`Text`, Sequence[str]
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/usage/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`, Literal['quantitative', 'ordinal', 'temporal', 'nominal']
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria:


        * ``"quantitative"`` is the default type if (1) the encoded field contains ``bin``
          or ``aggregate`` except ``"argmin"`` and ``"argmax"``, (2) the encoding channel is
          ``latitude`` or ``longitude`` channel or (3) if the specified scale type is `a
          quantitative scale <https://vega.github.io/vega-lite/docs/scale.html#type>`__.
        * ``"temporal"`` is the default type if (1) the encoded field contains ``timeUnit``
          or (2) the specified scale type is a time or utc scale
        * ``"ordinal"`` is the default type if (1) the encoded field contains a `custom sort
          order
          <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
          (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
          channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ):


        * ``"quantitative"`` if the datum is a number
        * ``"nominal"`` if the datum is a string
        * ``"temporal"`` if the datum is `a date time object
          <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:**


        * Data ``type`` describes the semantics of the data rather than the primitive data
          types (number, string, etc.). The same primitive data type can have different
          types of measurement. For example, numeric data can represent quantitative,
          ordinal, or nominal data.
        * Data values for a temporal field can be either a date-time string (e.g.,
          ``"2015-03-07 12:32:17"``, ``"17:01"``, ``"2015-03-16"``. ``"2015"`` ) or a
          timestamp number (e.g., ``1552199579097`` ).
        * When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
          ``type`` property can be either ``"quantitative"`` (for using a linear bin scale)
          or `"ordinal" (for using an ordinal bin scale)
          <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__.
        * When using with `timeUnit
          <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type`` property
          can be either ``"temporal"`` (default, for using a temporal scale) or `"ordinal"
          (for using an ordinal scale)
          <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__.
        * When using with `aggregate
          <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type`` property
          refers to the post-aggregation data type. For example, we can calculate count
          ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate": "distinct",
          "field": "cat"}``. The ``"type"`` of the aggregate output is ``"quantitative"``.
        * Secondary channels (e.g., ``x2``, ``y2``, ``xError``, ``yError`` ) do not have
          ``type`` as they must have exactly the same type as their primary channels (e.g.,
          ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _class_is_valid_at_instantiation = False
    _encoding_name = 'radius'

    @overload
    def aggregate(self, _: Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], **kwds) -> 'Radius':
        ...

    @overload
    def aggregate(self, argmax: Union[str, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def aggregate(self, argmin: Union[str, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def bandPosition(self, _: float, **kwds) -> 'Radius':
        ...

    @overload
    def bin(self, _: bool, **kwds) -> 'Radius':
        ...

    @overload
    def bin(self, anchor: Union[float, UndefinedType]=Undefined, base: Union[float, UndefinedType]=Undefined, binned: Union[bool, UndefinedType]=Undefined, divide: Union[Sequence[float], UndefinedType]=Undefined, extent: Union[dict, core._Parameter, core.SchemaBase, Sequence[float], UndefinedType]=Undefined, maxbins: Union[float, UndefinedType]=Undefined, minstep: Union[float, UndefinedType]=Undefined, nice: Union[bool, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, steps: Union[Sequence[float], UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def bin(self, _: str, **kwds) -> 'Radius':
        ...

    @overload
    def bin(self, _: None, **kwds) -> 'Radius':
        ...

    @overload
    def field(self, _: str, **kwds) -> 'Radius':
        ...

    @overload
    def field(self, repeat: Union[Literal['row', 'column', 'repeat', 'layer'], UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def scale(self, align: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, base: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, bins: Union[dict, core.SchemaBase, Sequence[float], UndefinedType]=Undefined, clamp: Union[bool, dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, constant: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, domain: Union[str, dict, core._Parameter, core.SchemaBase, Sequence[Union[str, bool, dict, None, float, core._Parameter, core.SchemaBase]], UndefinedType]=Undefined, domainMax: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, domainMid: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, domainMin: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, domainRaw: Union[dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, exponent: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, interpolate: Union[dict, core._Parameter, core.SchemaBase, Literal['rgb', 'lab', 'hcl', 'hsl', 'hsl-long', 'hcl-long', 'cubehelix', 'cubehelix-long'], UndefinedType]=Undefined, nice: Union[bool, dict, float, core._Parameter, core.SchemaBase, Literal['millisecond', 'second', 'minute', 'hour', 'day', 'week', 'month', 'year'], UndefinedType]=Undefined, padding: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, paddingInner: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, paddingOuter: Union[dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, range: Union[dict, core.SchemaBase, Sequence[Union[str, dict, float, core._Parameter, core.SchemaBase, Sequence[float]]], Literal['width', 'height', 'symbol', 'category', 'ordinal', 'ramp', 'diverging', 'heatmap'], UndefinedType]=Undefined, rangeMax: Union[str, dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, rangeMin: Union[str, dict, float, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, reverse: Union[bool, dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, round: Union[bool, dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, scheme: Union[dict, Sequence[str], core._Parameter, core.SchemaBase, Literal['rainbow', 'sinebow'], Literal['blues', 'tealblues', 'teals', 'greens', 'browns', 'greys', 'purples', 'warmgreys', 'reds', 'oranges'], Literal['accent', 'category10', 'category20', 'category20b', 'category20c', 'dark2', 'paired', 'pastel1', 'pastel2', 'set1', 'set2', 'set3', 'tableau10', 'tableau20'], Literal['blueorange', 'blueorange-3', 'blueorange-4', 'blueorange-5', 'blueorange-6', 'blueorange-7', 'blueorange-8', 'blueorange-9', 'blueorange-10', 'blueorange-11', 'brownbluegreen', 'brownbluegreen-3', 'brownbluegreen-4', 'brownbluegreen-5', 'brownbluegreen-6', 'brownbluegreen-7', 'brownbluegreen-8', 'brownbluegreen-9', 'brownbluegreen-10', 'brownbluegreen-11', 'purplegreen', 'purplegreen-3', 'purplegreen-4', 'purplegreen-5', 'purplegreen-6', 'purplegreen-7', 'purplegreen-8', 'purplegreen-9', 'purplegreen-10', 'purplegreen-11', 'pinkyellowgreen', 'pinkyellowgreen-3', 'pinkyellowgreen-4', 'pinkyellowgreen-5', 'pinkyellowgreen-6', 'pinkyellowgreen-7', 'pinkyellowgreen-8', 'pinkyellowgreen-9', 'pinkyellowgreen-10', 'pinkyellowgreen-11', 'purpleorange', 'purpleorange-3', 'purpleorange-4', 'purpleorange-5', 'purpleorange-6', 'purpleorange-7', 'purpleorange-8', 'purpleorange-9', 'purpleorange-10', 'purpleorange-11', 'redblue', 'redblue-3', 'redblue-4', 'redblue-5', 'redblue-6', 'redblue-7', 'redblue-8', 'redblue-9', 'redblue-10', 'redblue-11', 'redgrey', 'redgrey-3', 'redgrey-4', 'redgrey-5', 'redgrey-6', 'redgrey-7', 'redgrey-8', 'redgrey-9', 'redgrey-10', 'redgrey-11', 'redyellowblue', 'redyellowblue-3', 'redyellowblue-4', 'redyellowblue-5', 'redyellowblue-6', 'redyellowblue-7', 'redyellowblue-8', 'redyellowblue-9', 'redyellowblue-10', 'redyellowblue-11', 'redyellowgreen', 'redyellowgreen-3', 'redyellowgreen-4', 'redyellowgreen-5', 'redyellowgreen-6', 'redyellowgreen-7', 'redyellowgreen-8', 'redyellowgreen-9', 'redyellowgreen-10', 'redyellowgreen-11', 'spectral', 'spectral-3', 'spectral-4', 'spectral-5', 'spectral-6', 'spectral-7', 'spectral-8', 'spectral-9', 'spectral-10', 'spectral-11'], Literal['turbo', 'viridis', 'inferno', 'magma', 'plasma', 'cividis', 'bluegreen', 'bluegreen-3', 'bluegreen-4', 'bluegreen-5', 'bluegreen-6', 'bluegreen-7', 'bluegreen-8', 'bluegreen-9', 'bluepurple', 'bluepurple-3', 'bluepurple-4', 'bluepurple-5', 'bluepurple-6', 'bluepurple-7', 'bluepurple-8', 'bluepurple-9', 'goldgreen', 'goldgreen-3', 'goldgreen-4', 'goldgreen-5', 'goldgreen-6', 'goldgreen-7', 'goldgreen-8', 'goldgreen-9', 'goldorange', 'goldorange-3', 'goldorange-4', 'goldorange-5', 'goldorange-6', 'goldorange-7', 'goldorange-8', 'goldorange-9', 'goldred', 'goldred-3', 'goldred-4', 'goldred-5', 'goldred-6', 'goldred-7', 'goldred-8', 'goldred-9', 'greenblue', 'greenblue-3', 'greenblue-4', 'greenblue-5', 'greenblue-6', 'greenblue-7', 'greenblue-8', 'greenblue-9', 'orangered', 'orangered-3', 'orangered-4', 'orangered-5', 'orangered-6', 'orangered-7', 'orangered-8', 'orangered-9', 'purplebluegreen', 'purplebluegreen-3', 'purplebluegreen-4', 'purplebluegreen-5', 'purplebluegreen-6', 'purplebluegreen-7', 'purplebluegreen-8', 'purplebluegreen-9', 'purpleblue', 'purpleblue-3', 'purpleblue-4', 'purpleblue-5', 'purpleblue-6', 'purpleblue-7', 'purpleblue-8', 'purpleblue-9', 'purplered', 'purplered-3', 'purplered-4', 'purplered-5', 'purplered-6', 'purplered-7', 'purplered-8', 'purplered-9', 'redpurple', 'redpurple-3', 'redpurple-4', 'redpurple-5', 'redpurple-6', 'redpurple-7', 'redpurple-8', 'redpurple-9', 'yellowgreenblue', 'yellowgreenblue-3', 'yellowgreenblue-4', 'yellowgreenblue-5', 'yellowgreenblue-6', 'yellowgreenblue-7', 'yellowgreenblue-8', 'yellowgreenblue-9', 'yellowgreen', 'yellowgreen-3', 'yellowgreen-4', 'yellowgreen-5', 'yellowgreen-6', 'yellowgreen-7', 'yellowgreen-8', 'yellowgreen-9', 'yelloworangebrown', 'yelloworangebrown-3', 'yelloworangebrown-4', 'yelloworangebrown-5', 'yelloworangebrown-6', 'yelloworangebrown-7', 'yelloworangebrown-8', 'yelloworangebrown-9', 'yelloworangered', 'yelloworangered-3', 'yelloworangered-4', 'yelloworangered-5', 'yelloworangered-6', 'yelloworangered-7', 'yelloworangered-8', 'yelloworangered-9', 'darkblue', 'darkblue-3', 'darkblue-4', 'darkblue-5', 'darkblue-6', 'darkblue-7', 'darkblue-8', 'darkblue-9', 'darkgold', 'darkgold-3', 'darkgold-4', 'darkgold-5', 'darkgold-6', 'darkgold-7', 'darkgold-8', 'darkgold-9', 'darkgreen', 'darkgreen-3', 'darkgreen-4', 'darkgreen-5', 'darkgreen-6', 'darkgreen-7', 'darkgreen-8', 'darkgreen-9', 'darkmulti', 'darkmulti-3', 'darkmulti-4', 'darkmulti-5', 'darkmulti-6', 'darkmulti-7', 'darkmulti-8', 'darkmulti-9', 'darkred', 'darkred-3', 'darkred-4', 'darkred-5', 'darkred-6', 'darkred-7', 'darkred-8', 'darkred-9', 'lightgreyred', 'lightgreyred-3', 'lightgreyred-4', 'lightgreyred-5', 'lightgreyred-6', 'lightgreyred-7', 'lightgreyred-8', 'lightgreyred-9', 'lightgreyteal', 'lightgreyteal-3', 'lightgreyteal-4', 'lightgreyteal-5', 'lightgreyteal-6', 'lightgreyteal-7', 'lightgreyteal-8', 'lightgreyteal-9', 'lightmulti', 'lightmulti-3', 'lightmulti-4', 'lightmulti-5', 'lightmulti-6', 'lightmulti-7', 'lightmulti-8', 'lightmulti-9', 'lightorange', 'lightorange-3', 'lightorange-4', 'lightorange-5', 'lightorange-6', 'lightorange-7', 'lightorange-8', 'lightorange-9', 'lighttealblue', 'lighttealblue-3', 'lighttealblue-4', 'lighttealblue-5', 'lighttealblue-6', 'lighttealblue-7', 'lighttealblue-8', 'lighttealblue-9'], UndefinedType]=Undefined, type: Union[core.SchemaBase, Literal['linear', 'log', 'pow', 'sqrt', 'symlog', 'identity', 'sequential', 'time', 'utc', 'quantile', 'quantize', 'threshold', 'bin-ordinal', 'ordinal', 'point', 'band'], UndefinedType]=Undefined, zero: Union[bool, dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def scale(self, _: None, **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: List[float], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: List[str], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: List[bool], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: List[core.DateTime], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: Literal['ascending', 'descending'], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: Literal['x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity', 'strokeOpacity', 'opacity', 'text'], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: Literal['-x', '-y', '-color', '-fill', '-stroke', '-strokeWidth', '-size', '-shape', '-fillOpacity', '-strokeOpacity', '-opacity', '-text'], **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, field: Union[str, dict, core.SchemaBase, UndefinedType]=Undefined, op: Union[core.SchemaBase, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, order: Union[None, core.SchemaBase, Literal['ascending', 'descending'], UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, encoding: Union[core.SchemaBase, Literal['x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity', 'strokeOpacity', 'opacity', 'text'], UndefinedType]=Undefined, order: Union[None, core.SchemaBase, Literal['ascending', 'descending'], UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def sort(self, _: None, **kwds) -> 'Radius':
        ...

    @overload
    def stack(self, _: Literal['zero', 'center', 'normalize'], **kwds) -> 'Radius':
        ...

    @overload
    def stack(self, _: None, **kwds) -> 'Radius':
        ...

    @overload
    def stack(self, _: bool, **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, _: Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, _: Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, _: Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, _: Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, _: Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, _: Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], **kwds) -> 'Radius':
        ...

    @overload
    def timeUnit(self, binned: Union[bool, UndefinedType]=Undefined, maxbins: Union[float, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, unit: Union[core.SchemaBase, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], UndefinedType]=Undefined, utc: Union[bool, UndefinedType]=Undefined, **kwds) -> 'Radius':
        ...

    @overload
    def title(self, _: str, **kwds) -> 'Radius':
        ...

    @overload
    def title(self, _: List[str], **kwds) -> 'Radius':
        ...

    @overload
    def title(self, _: None, **kwds) -> 'Radius':
        ...

    @overload
    def type(self, _: Literal['quantitative', 'ordinal', 'temporal', 'nominal'], **kwds) -> 'Radius':
        ...

    def __init__(self, shorthand: Union[str, dict, Sequence[str], core.SchemaBase, UndefinedType]=Undefined, aggregate: Union[dict, core.SchemaBase, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, bandPosition: Union[float, UndefinedType]=Undefined, bin: Union[str, bool, dict, None, core.SchemaBase, UndefinedType]=Undefined, field: Union[str, dict, core.SchemaBase, UndefinedType]=Undefined, scale: Union[dict, None, core.SchemaBase, UndefinedType]=Undefined, sort: Union[dict, None, Sequence[str], Sequence[bool], core.SchemaBase, Sequence[float], Literal['ascending', 'descending'], Sequence[Union[dict, core.SchemaBase]], Literal['x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity', 'strokeOpacity', 'opacity', 'text'], Literal['-x', '-y', '-color', '-fill', '-stroke', '-strokeWidth', '-size', '-shape', '-fillOpacity', '-strokeOpacity', '-opacity', '-text'], UndefinedType]=Undefined, stack: Union[bool, None, core.SchemaBase, Literal['zero', 'center', 'normalize'], UndefinedType]=Undefined, timeUnit: Union[dict, core.SchemaBase, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], UndefinedType]=Undefined, title: Union[str, None, Sequence[str], core.SchemaBase, UndefinedType]=Undefined, type: Union[core.SchemaBase, Literal['quantitative', 'ordinal', 'temporal', 'nominal'], UndefinedType]=Undefined, **kwds):
        super(Radius, self).__init__(shorthand=shorthand, aggregate=aggregate, bandPosition=bandPosition, bin=bin, field=field, scale=scale, sort=sort, stack=stack, timeUnit=timeUnit, title=title, type=type, **kwds)
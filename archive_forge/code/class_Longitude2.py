import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
@with_property_setters
class Longitude2(FieldChannelMixin, core.SecondaryFieldDef):
    """Longitude2 schema wrapper
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

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
    bin : None
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
    """
    _class_is_valid_at_instantiation = False
    _encoding_name = 'longitude2'

    @overload
    def aggregate(self, _: Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], **kwds) -> 'Longitude2':
        ...

    @overload
    def aggregate(self, argmax: Union[str, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Longitude2':
        ...

    @overload
    def aggregate(self, argmin: Union[str, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Longitude2':
        ...

    @overload
    def bandPosition(self, _: float, **kwds) -> 'Longitude2':
        ...

    @overload
    def bin(self, _: None, **kwds) -> 'Longitude2':
        ...

    @overload
    def field(self, _: str, **kwds) -> 'Longitude2':
        ...

    @overload
    def field(self, repeat: Union[Literal['row', 'column', 'repeat', 'layer'], UndefinedType]=Undefined, **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, _: Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, _: Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, _: Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, _: Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, _: Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, _: Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], **kwds) -> 'Longitude2':
        ...

    @overload
    def timeUnit(self, binned: Union[bool, UndefinedType]=Undefined, maxbins: Union[float, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, unit: Union[core.SchemaBase, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], UndefinedType]=Undefined, utc: Union[bool, UndefinedType]=Undefined, **kwds) -> 'Longitude2':
        ...

    @overload
    def title(self, _: str, **kwds) -> 'Longitude2':
        ...

    @overload
    def title(self, _: List[str], **kwds) -> 'Longitude2':
        ...

    @overload
    def title(self, _: None, **kwds) -> 'Longitude2':
        ...

    def __init__(self, shorthand: Union[str, dict, Sequence[str], core.SchemaBase, UndefinedType]=Undefined, aggregate: Union[dict, core.SchemaBase, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, bandPosition: Union[float, UndefinedType]=Undefined, bin: Union[None, UndefinedType]=Undefined, field: Union[str, dict, core.SchemaBase, UndefinedType]=Undefined, timeUnit: Union[dict, core.SchemaBase, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], UndefinedType]=Undefined, title: Union[str, None, Sequence[str], core.SchemaBase, UndefinedType]=Undefined, **kwds):
        super(Longitude2, self).__init__(shorthand=shorthand, aggregate=aggregate, bandPosition=bandPosition, bin=bin, field=field, timeUnit=timeUnit, title=title, **kwds)
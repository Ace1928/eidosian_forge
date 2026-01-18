from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ScaleDatumDef(OffsetDef):
    """ScaleDatumDef schema wrapper

    Parameters
    ----------

    bandPosition : float
        Relative position on a band of a stacked, binned, time unit, or band scale. For
        example, the marks will be positioned at the beginning of the band if set to ``0``,
        and at the middle of the band if set to ``0.5``.
    datum : str, bool, dict, None, float, :class:`ExprRef`, :class:`DateTime`, :class:`RepeatRef`, :class:`PrimitiveValue`
        A constant value in data domain.
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
    type : :class:`Type`, Literal['quantitative', 'ordinal', 'temporal', 'nominal', 'geojson']
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
    _schema = {'$ref': '#/definitions/ScaleDatumDef'}

    def __init__(self, bandPosition: Union[float, UndefinedType]=Undefined, datum: Union[str, bool, dict, None, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, scale: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, title: Union[str, None, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, type: Union['SchemaBase', Literal['quantitative', 'ordinal', 'temporal', 'nominal', 'geojson'], UndefinedType]=Undefined, **kwds):
        super(ScaleDatumDef, self).__init__(bandPosition=bandPosition, datum=datum, scale=scale, title=title, type=type, **kwds)
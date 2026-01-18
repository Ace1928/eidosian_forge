from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FormatConfig(VegaLiteSchema):
    """FormatConfig schema wrapper

    Parameters
    ----------

    normalizedNumberFormat : str
        If normalizedNumberFormatType is not specified, D3 number format for axis labels,
        text marks, and tooltips of normalized stacked fields (fields with ``stack:
        "normalize"`` ). For example ``"s"`` for SI units. Use `D3's number format pattern
        <https://github.com/d3/d3-format#locale_format>`__.

        If ``config.normalizedNumberFormatType`` is specified and
        ``config.customFormatTypes`` is ``true``, this value will be passed as ``format``
        alongside ``datum.value`` to the ``config.numberFormatType`` function. **Default
        value:** ``%``
    normalizedNumberFormatType : str
        `Custom format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__ for
        ``config.normalizedNumberFormat``.

        **Default value:** ``undefined`` -- This is equilvalent to call D3-format, which is
        exposed as `format in Vega-Expression
        <https://vega.github.io/vega/docs/expressions/#format>`__. **Note:** You must also
        set ``customFormatTypes`` to ``true`` to use this feature.
    numberFormat : str
        If numberFormatType is not specified, D3 number format for guide labels, text marks,
        and tooltips of non-normalized fields (fields *without* ``stack: "normalize"`` ).
        For example ``"s"`` for SI units. Use `D3's number format pattern
        <https://github.com/d3/d3-format#locale_format>`__.

        If ``config.numberFormatType`` is specified and ``config.customFormatTypes`` is
        ``true``, this value will be passed as ``format`` alongside ``datum.value`` to the
        ``config.numberFormatType`` function.
    numberFormatType : str
        `Custom format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__ for
        ``config.numberFormat``.

        **Default value:** ``undefined`` -- This is equilvalent to call D3-format, which is
        exposed as `format in Vega-Expression
        <https://vega.github.io/vega/docs/expressions/#format>`__. **Note:** You must also
        set ``customFormatTypes`` to ``true`` to use this feature.
    timeFormat : str
        Default time format for raw time values (without time units) in text marks, legend
        labels and header labels.

        **Default value:** ``"%b %d, %Y"`` **Note:** Axes automatically determine the format
        for each label automatically so this config does not affect axes.
    timeFormatType : str
        `Custom format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__ for
        ``config.timeFormat``.

        **Default value:** ``undefined`` -- This is equilvalent to call D3-time-format,
        which is exposed as `timeFormat in Vega-Expression
        <https://vega.github.io/vega/docs/expressions/#timeFormat>`__. **Note:** You must
        also set ``customFormatTypes`` to ``true`` and there must *not* be a ``timeUnit``
        defined to use this feature.
    """
    _schema = {'$ref': '#/definitions/FormatConfig'}

    def __init__(self, normalizedNumberFormat: Union[str, UndefinedType]=Undefined, normalizedNumberFormatType: Union[str, UndefinedType]=Undefined, numberFormat: Union[str, UndefinedType]=Undefined, numberFormatType: Union[str, UndefinedType]=Undefined, timeFormat: Union[str, UndefinedType]=Undefined, timeFormatType: Union[str, UndefinedType]=Undefined, **kwds):
        super(FormatConfig, self).__init__(normalizedNumberFormat=normalizedNumberFormat, normalizedNumberFormatType=normalizedNumberFormatType, numberFormat=numberFormat, numberFormatType=numberFormatType, timeFormat=timeFormat, timeFormatType=timeFormatType, **kwds)
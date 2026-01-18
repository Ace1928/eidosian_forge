from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TopLevelSelectionParameter(TopLevelParameter):
    """TopLevelSelectionParameter schema wrapper

    Parameters
    ----------

    name : str, :class:`ParameterName`
        Required. A unique name for the selection parameter. Selection names should be valid
        JavaScript identifiers: they should contain only alphanumeric characters (or "$", or
        "_") and may not start with a digit. Reserved keywords that may not be used as
        parameter names are "datum", "event", "item", and "parent".
    select : dict, :class:`SelectionType`, Literal['point', 'interval'], :class:`PointSelectionConfig`, :class:`IntervalSelectionConfig`
        Determines the default event processing and data query for the selection. Vega-Lite
        currently supports two selection types:


        * ``"point"`` -- to select multiple discrete data values; the first value is
          selected on ``click`` and additional values toggled on shift-click.
        * ``"interval"`` -- to select a continuous range of data values on ``drag``.
    bind : str, dict, :class:`Binding`, :class:`BindInput`, :class:`BindRange`, :class:`BindDirect`, :class:`BindCheckbox`, :class:`LegendBinding`, :class:`BindRadioSelect`, :class:`LegendStreamBinding`
        When set, a selection is populated by input elements (also known as dynamic query
        widgets) or by interacting with the corresponding legend. Direct manipulation
        interaction is disabled by default; to re-enable it, set the selection's `on
        <https://vega.github.io/vega-lite/docs/selection.html#common-selection-properties>`__
        property.

        Legend bindings are restricted to selections that only specify a single field or
        encoding.

        Query widget binding takes the form of Vega's `input element binding definition
        <https://vega.github.io/vega/docs/signals/#bind>`__ or can be a mapping between
        projected field/encodings and binding definitions.

        **See also:** `bind <https://vega.github.io/vega-lite/docs/bind.html>`__
        documentation.
    value : str, bool, dict, None, float, :class:`DateTime`, :class:`SelectionInit`, :class:`PrimitiveValue`, :class:`SelectionInitIntervalMapping`, Sequence[dict, :class:`SelectionInitMapping`]
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/selection.html#project>`__ and initial
        values.

        **See also:** `init <https://vega.github.io/vega-lite/docs/value.html>`__
        documentation.
    views : Sequence[str]
        By default, top-level selections are applied to every view in the visualization. If
        this property is specified, selections will only be applied to views with the given
        names.
    """
    _schema = {'$ref': '#/definitions/TopLevelSelectionParameter'}

    def __init__(self, name: Union[str, 'SchemaBase', UndefinedType]=Undefined, select: Union[dict, 'SchemaBase', Literal['point', 'interval'], UndefinedType]=Undefined, bind: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, value: Union[str, bool, dict, None, float, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, views: Union[Sequence[str], UndefinedType]=Undefined, **kwds):
        super(TopLevelSelectionParameter, self).__init__(name=name, select=select, bind=bind, value=value, views=views, **kwds)
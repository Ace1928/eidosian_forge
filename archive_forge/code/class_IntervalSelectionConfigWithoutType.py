from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class IntervalSelectionConfigWithoutType(VegaLiteSchema):
    """IntervalSelectionConfigWithoutType schema wrapper

    Parameters
    ----------

    clear : str, bool, dict, :class:`Stream`, :class:`EventStream`, :class:`MergedStream`, :class:`DerivedStream`
        Clears the selection, emptying it of all values. This property can be a `Event
        Stream <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable
        clear.

        **Default value:** ``dblclick``.

        **See also:** `clear examples
        <https://vega.github.io/vega-lite/docs/selection.html#clear>`__ in the
        documentation.
    encodings : Sequence[:class:`SingleDefUnitChannel`, Literal['x', 'y', 'xOffset', 'yOffset', 'x2', 'y2', 'longitude', 'latitude', 'longitude2', 'latitude2', 'theta', 'theta2', 'radius', 'radius2', 'color', 'fill', 'stroke', 'opacity', 'fillOpacity', 'strokeOpacity', 'strokeWidth', 'strokeDash', 'size', 'angle', 'shape', 'key', 'text', 'href', 'url', 'description']]
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** The `projection with encodings and fields section
        <https://vega.github.io/vega-lite/docs/selection.html#project>`__ in the
        documentation.
    fields : Sequence[str, :class:`FieldName`]
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** The `projection with encodings and fields section
        <https://vega.github.io/vega-lite/docs/selection.html#project>`__ in the
        documentation.
    mark : dict, :class:`BrushConfig`
        An interval selection also adds a rectangle mark to depict the extents of the
        interval. The ``mark`` property can be used to customize the appearance of the mark.

        **See also:** `mark examples
        <https://vega.github.io/vega-lite/docs/selection.html#mark>`__ in the documentation.
    on : str, dict, :class:`Stream`, :class:`EventStream`, :class:`MergedStream`, :class:`DerivedStream`
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.

        **See also:** `on examples
        <https://vega.github.io/vega-lite/docs/selection.html#on>`__ in the documentation.
    resolve : :class:`SelectionResolution`, Literal['global', 'union', 'intersect']
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        One of:


        * ``"global"`` -- only one brush exists for the entire SPLOM. When the user begins
          to drag, any previous brushes are cleared, and a new one is constructed.
        * ``"union"`` -- each cell contains its own brush, and points are highlighted if
          they lie within *any* of these individual brushes.
        * ``"intersect"`` -- each cell contains its own brush, and points are highlighted
          only if they fall within *all* of these individual brushes.

        **Default value:** ``global``.

        **See also:** `resolve examples
        <https://vega.github.io/vega-lite/docs/selection.html#resolve>`__ in the
        documentation.
    translate : str, bool
        When truthy, allows a user to interactively move an interval selection
        back-and-forth. Can be ``true``, ``false`` (to disable panning), or a `Vega event
        stream definition <https://vega.github.io/vega/docs/event-streams/>`__ which must
        include a start and end event to trigger continuous panning. Discrete panning (e.g.,
        pressing the left/right arrow keys) will be supported in future versions.

        **Default value:** ``true``, which corresponds to ``[pointerdown, window:pointerup]
        > window:pointermove!``. This default allows users to clicks and drags within an
        interval selection to reposition it.

        **See also:** `translate examples
        <https://vega.github.io/vega-lite/docs/selection.html#translate>`__ in the
        documentation.
    zoom : str, bool
        When truthy, allows a user to interactively resize an interval selection. Can be
        ``true``, ``false`` (to disable zooming), or a `Vega event stream definition
        <https://vega.github.io/vega/docs/event-streams/>`__. Currently, only ``wheel``
        events are supported, but custom event streams can still be used to specify filters,
        debouncing, and throttling. Future versions will expand the set of events that can
        trigger this transformation.

        **Default value:** ``true``, which corresponds to ``wheel!``. This default allows
        users to use the mouse wheel to resize an interval selection.

        **See also:** `zoom examples
        <https://vega.github.io/vega-lite/docs/selection.html#zoom>`__ in the documentation.
    """
    _schema = {'$ref': '#/definitions/IntervalSelectionConfigWithoutType'}

    def __init__(self, clear: Union[str, bool, dict, 'SchemaBase', UndefinedType]=Undefined, encodings: Union[Sequence[Union['SchemaBase', Literal['x', 'y', 'xOffset', 'yOffset', 'x2', 'y2', 'longitude', 'latitude', 'longitude2', 'latitude2', 'theta', 'theta2', 'radius', 'radius2', 'color', 'fill', 'stroke', 'opacity', 'fillOpacity', 'strokeOpacity', 'strokeWidth', 'strokeDash', 'size', 'angle', 'shape', 'key', 'text', 'href', 'url', 'description']]], UndefinedType]=Undefined, fields: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, mark: Union[dict, 'SchemaBase', UndefinedType]=Undefined, on: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, resolve: Union['SchemaBase', Literal['global', 'union', 'intersect'], UndefinedType]=Undefined, translate: Union[str, bool, UndefinedType]=Undefined, zoom: Union[str, bool, UndefinedType]=Undefined, **kwds):
        super(IntervalSelectionConfigWithoutType, self).__init__(clear=clear, encodings=encodings, fields=fields, mark=mark, on=on, resolve=resolve, translate=translate, zoom=zoom, **kwds)
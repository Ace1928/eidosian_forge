from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class PointSelectionConfigWithoutType(VegaLiteSchema):
    """PointSelectionConfigWithoutType schema wrapper

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
    nearest : bool
        When true, an invisible voronoi diagram is computed to accelerate discrete
        selection. The data value *nearest* the mouse cursor is added to the selection.

        **Default value:** ``false``, which means that data values must be interacted with
        directly (e.g., clicked on) to be added to the selection.

        **See also:** `nearest examples
        <https://vega.github.io/vega-lite/docs/selection.html#nearest>`__ documentation.
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
    toggle : str, bool
        Controls whether data values should be toggled (inserted or removed from a point
        selection) or only ever inserted into point selections.

        One of:


        * ``true`` -- the default behavior, which corresponds to ``"event.shiftKey"``.  As a
          result, data values are toggled when the user interacts with the shift-key
          pressed.
        * ``false`` -- disables toggling behaviour; the selection will only ever contain a
          single data value corresponding to the most recent interaction.
        * A `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ which is
          re-evaluated as the user interacts. If the expression evaluates to ``true``, the
          data value is toggled into or out of the point selection. If the expression
          evaluates to ``false``, the point selection is first cleared, and the data value
          is then inserted. For example, setting the value to the Vega expression ``"true"``
          will toggle data values without the user pressing the shift-key.

        **Default value:** ``true``

        **See also:** `toggle examples
        <https://vega.github.io/vega-lite/docs/selection.html#toggle>`__ in the
        documentation.
    """
    _schema = {'$ref': '#/definitions/PointSelectionConfigWithoutType'}

    def __init__(self, clear: Union[str, bool, dict, 'SchemaBase', UndefinedType]=Undefined, encodings: Union[Sequence[Union['SchemaBase', Literal['x', 'y', 'xOffset', 'yOffset', 'x2', 'y2', 'longitude', 'latitude', 'longitude2', 'latitude2', 'theta', 'theta2', 'radius', 'radius2', 'color', 'fill', 'stroke', 'opacity', 'fillOpacity', 'strokeOpacity', 'strokeWidth', 'strokeDash', 'size', 'angle', 'shape', 'key', 'text', 'href', 'url', 'description']]], UndefinedType]=Undefined, fields: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, nearest: Union[bool, UndefinedType]=Undefined, on: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, resolve: Union['SchemaBase', Literal['global', 'union', 'intersect'], UndefinedType]=Undefined, toggle: Union[str, bool, UndefinedType]=Undefined, **kwds):
        super(PointSelectionConfigWithoutType, self).__init__(clear=clear, encodings=encodings, fields=fields, nearest=nearest, on=on, resolve=resolve, toggle=toggle, **kwds)
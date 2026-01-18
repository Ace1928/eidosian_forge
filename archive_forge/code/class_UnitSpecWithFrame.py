from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class UnitSpecWithFrame(VegaLiteSchema):
    """UnitSpecWithFrame schema wrapper

    Parameters
    ----------

    mark : str, dict, :class:`Mark`, :class:`AnyMark`, :class:`BoxPlot`, :class:`MarkDef`, :class:`ErrorBar`, :class:`ErrorBand`, :class:`BoxPlotDef`, :class:`ErrorBarDef`, :class:`ErrorBandDef`, :class:`CompositeMark`, :class:`CompositeMarkDef`, Literal['arc', 'area', 'bar', 'image', 'line', 'point', 'rect', 'rule', 'text', 'tick', 'trail', 'circle', 'square', 'geoshape']
        A string describing the mark type (one of ``"bar"``, ``"circle"``, ``"square"``,
        ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"rule"``, ``"geoshape"``, and
        ``"text"`` ) or a `mark definition object
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.
    data : dict, None, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : str
        Description of this mark for commenting purpose.
    encoding : dict, :class:`Encoding`
        A key-value mapping between encoding channels and definition of fields.
    height : str, dict, float, :class:`Step`
        The height of a visualization.


        * For a plot with a continuous y-field, height should be a number.
        * For a plot with either a discrete y-field or no y-field, height can be either a
          number indicating a fixed height or an object in the form of ``{step: number}``
          defining the height per discrete step. (No y-field is equivalent to having one
          discrete step.)
        * To enable responsive sizing on height, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousHeight`` for a plot with a
        continuous y-field and ``config.view.discreteHeight`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        height of a single view and the ``"container"`` option cannot be used.

        **See also:** `height <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    name : str
        Name of the visualization for later reference.
    params : Sequence[dict, :class:`SelectionParameter`]
        An array of parameters that may either be simple variables, or more complex
        selections that map user input to data queries.
    projection : dict, :class:`Projection`
        An object defining properties of geographic projection, which will be applied to
        ``shape`` path for ``"geoshape"`` marks and to ``latitude`` and ``"longitude"``
        channels for other marks.
    title : str, dict, :class:`Text`, Sequence[str], :class:`TitleParams`
        Title for the plot.
    transform : Sequence[dict, :class:`Transform`, :class:`BinTransform`, :class:`FoldTransform`, :class:`LoessTransform`, :class:`PivotTransform`, :class:`StackTransform`, :class:`ExtentTransform`, :class:`FilterTransform`, :class:`ImputeTransform`, :class:`LookupTransform`, :class:`SampleTransform`, :class:`WindowTransform`, :class:`DensityTransform`, :class:`FlattenTransform`, :class:`QuantileTransform`, :class:`TimeUnitTransform`, :class:`AggregateTransform`, :class:`CalculateTransform`, :class:`RegressionTransform`, :class:`JoinAggregateTransform`]
        An array of data transformations such as filter and new field calculation.
    view : dict, :class:`ViewBackground`
        An object defining the view background's fill and stroke.

        **Default value:** none (transparent)
    width : str, dict, float, :class:`Step`
        The width of a visualization.


        * For a plot with a continuous x-field, width should be a number.
        * For a plot with either a discrete x-field or no x-field, width can be either a
          number indicating a fixed width or an object in the form of ``{step: number}``
          defining the width per discrete step. (No x-field is equivalent to having one
          discrete step.)
        * To enable responsive sizing on width, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousWidth`` for a plot with a
        continuous x-field and ``config.view.discreteWidth`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        width of a single view and the ``"container"`` option cannot be used.

        **See also:** `width <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/UnitSpecWithFrame'}

    def __init__(self, mark: Union[str, dict, 'SchemaBase', Literal['arc', 'area', 'bar', 'image', 'line', 'point', 'rect', 'rule', 'text', 'tick', 'trail', 'circle', 'square', 'geoshape'], UndefinedType]=Undefined, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, encoding: Union[dict, 'SchemaBase', UndefinedType]=Undefined, height: Union[str, dict, float, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, params: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, projection: Union[dict, 'SchemaBase', UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, view: Union[dict, 'SchemaBase', UndefinedType]=Undefined, width: Union[str, dict, float, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(UnitSpecWithFrame, self).__init__(mark=mark, data=data, description=description, encoding=encoding, height=height, name=name, params=params, projection=projection, title=title, transform=transform, view=view, width=width, **kwds)
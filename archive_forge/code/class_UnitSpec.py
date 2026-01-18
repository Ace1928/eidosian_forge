from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class UnitSpec(VegaLiteSchema):
    """UnitSpec schema wrapper
    Base interface for a unit (single-view) specification.

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
    """
    _schema = {'$ref': '#/definitions/UnitSpec'}

    def __init__(self, mark: Union[str, dict, 'SchemaBase', Literal['arc', 'area', 'bar', 'image', 'line', 'point', 'rect', 'rule', 'text', 'tick', 'trail', 'circle', 'square', 'geoshape'], UndefinedType]=Undefined, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, encoding: Union[dict, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, params: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, projection: Union[dict, 'SchemaBase', UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(UnitSpec, self).__init__(mark=mark, data=data, description=description, encoding=encoding, name=name, params=params, projection=projection, title=title, transform=transform, **kwds)
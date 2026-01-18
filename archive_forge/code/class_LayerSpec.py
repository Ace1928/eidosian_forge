from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LayerSpec(Spec, NonNormalizedSpec):
    """LayerSpec schema wrapper
    A full layered plot specification, which may contains ``encoding`` and ``projection``
    properties that will be applied to underlying unit (single-view) specifications.

    Parameters
    ----------

    layer : Sequence[dict, :class:`UnitSpec`, :class:`LayerSpec`]
        Layer or single view specifications to be layered.

        **Note** : Specifications inside ``layer`` cannot use ``row`` and ``column``
        channels as layering facet specifications is not allowed. Instead, use the `facet
        operator <https://vega.github.io/vega-lite/docs/facet.html>`__ and place a layer
        inside a facet.
    data : dict, None, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : str
        Description of this mark for commenting purpose.
    encoding : dict, :class:`SharedEncoding`
        A shared key-value mapping between encoding channels and definition of fields in the
        underlying layers.
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
    projection : dict, :class:`Projection`
        An object defining properties of the geographic projection shared by underlying
        layers.
    resolve : dict, :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
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
    _schema = {'$ref': '#/definitions/LayerSpec'}

    def __init__(self, layer: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, encoding: Union[dict, 'SchemaBase', UndefinedType]=Undefined, height: Union[str, dict, float, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, projection: Union[dict, 'SchemaBase', UndefinedType]=Undefined, resolve: Union[dict, 'SchemaBase', UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, view: Union[dict, 'SchemaBase', UndefinedType]=Undefined, width: Union[str, dict, float, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(LayerSpec, self).__init__(layer=layer, data=data, description=description, encoding=encoding, height=height, name=name, projection=projection, resolve=resolve, title=title, transform=transform, view=view, width=width, **kwds)
from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SharedEncoding(VegaLiteSchema):
    """SharedEncoding schema wrapper

    Parameters
    ----------

    angle : dict

    color : dict

    description : dict

    detail : dict, :class:`FieldDefWithoutScale`, Sequence[dict, :class:`FieldDefWithoutScale`]
        Additional levels of detail for grouping data in aggregate views and in line, trail,
        and area marks without mapping data to a specific visual channel.
    fill : dict

    fillOpacity : dict

    href : dict

    key : dict

    latitude : dict

    latitude2 : dict

    longitude : dict

    longitude2 : dict

    opacity : dict

    order : dict, :class:`OrderOnlyDef`, :class:`OrderFieldDef`, :class:`OrderValueDef`, Sequence[dict, :class:`OrderFieldDef`]
        Order of the marks.


        * For stacked marks, this ``order`` channel encodes `stack order
          <https://vega.github.io/vega-lite/docs/stack.html#order>`__.
        * For line and trail marks, this ``order`` channel encodes order of data points in
          the lines. This can be useful for creating `a connected scatterplot
          <https://vega.github.io/vega-lite/examples/connected_scatterplot.html>`__. Setting
          ``order`` to ``{"value": null}`` makes the line marks use the original order in
          the data sources.
        * Otherwise, this ``order`` channel encodes layer order of the marks.

        **Note** : In aggregate plots, ``order`` field should be ``aggregate`` d to avoid
        creating additional aggregation grouping.
    radius : dict

    radius2 : dict

    shape : dict

    size : dict

    stroke : dict

    strokeDash : dict

    strokeOpacity : dict

    strokeWidth : dict

    text : dict

    theta : dict

    theta2 : dict

    tooltip : dict, None, :class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`, Sequence[dict, :class:`StringFieldDef`]
        The tooltip text to show upon mouse hover. Specifying ``tooltip`` encoding overrides
        `the tooltip property in the mark definition
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip in Vega-Lite.
    url : dict

    x : dict

    x2 : dict

    xError : dict

    xError2 : dict

    xOffset : dict

    y : dict

    y2 : dict

    yError : dict

    yError2 : dict

    yOffset : dict

    """
    _schema = {'$ref': '#/definitions/SharedEncoding'}

    def __init__(self, angle: Union[dict, UndefinedType]=Undefined, color: Union[dict, UndefinedType]=Undefined, description: Union[dict, UndefinedType]=Undefined, detail: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, fill: Union[dict, UndefinedType]=Undefined, fillOpacity: Union[dict, UndefinedType]=Undefined, href: Union[dict, UndefinedType]=Undefined, key: Union[dict, UndefinedType]=Undefined, latitude: Union[dict, UndefinedType]=Undefined, latitude2: Union[dict, UndefinedType]=Undefined, longitude: Union[dict, UndefinedType]=Undefined, longitude2: Union[dict, UndefinedType]=Undefined, opacity: Union[dict, UndefinedType]=Undefined, order: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, radius: Union[dict, UndefinedType]=Undefined, radius2: Union[dict, UndefinedType]=Undefined, shape: Union[dict, UndefinedType]=Undefined, size: Union[dict, UndefinedType]=Undefined, stroke: Union[dict, UndefinedType]=Undefined, strokeDash: Union[dict, UndefinedType]=Undefined, strokeOpacity: Union[dict, UndefinedType]=Undefined, strokeWidth: Union[dict, UndefinedType]=Undefined, text: Union[dict, UndefinedType]=Undefined, theta: Union[dict, UndefinedType]=Undefined, theta2: Union[dict, UndefinedType]=Undefined, tooltip: Union[dict, None, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, url: Union[dict, UndefinedType]=Undefined, x: Union[dict, UndefinedType]=Undefined, x2: Union[dict, UndefinedType]=Undefined, xError: Union[dict, UndefinedType]=Undefined, xError2: Union[dict, UndefinedType]=Undefined, xOffset: Union[dict, UndefinedType]=Undefined, y: Union[dict, UndefinedType]=Undefined, y2: Union[dict, UndefinedType]=Undefined, yError: Union[dict, UndefinedType]=Undefined, yError2: Union[dict, UndefinedType]=Undefined, yOffset: Union[dict, UndefinedType]=Undefined, **kwds):
        super(SharedEncoding, self).__init__(angle=angle, color=color, description=description, detail=detail, fill=fill, fillOpacity=fillOpacity, href=href, key=key, latitude=latitude, latitude2=latitude2, longitude=longitude, longitude2=longitude2, opacity=opacity, order=order, radius=radius, radius2=radius2, shape=shape, size=size, stroke=stroke, strokeDash=strokeDash, strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, text=text, theta=theta, theta2=theta2, tooltip=tooltip, url=url, x=x, x2=x2, xError=xError, xError2=xError2, xOffset=xOffset, y=y, y2=y2, yError=yError, yError2=yError2, yOffset=yOffset, **kwds)
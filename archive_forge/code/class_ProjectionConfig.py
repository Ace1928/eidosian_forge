from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ProjectionConfig(VegaLiteSchema):
    """ProjectionConfig schema wrapper

    Parameters
    ----------

    center : dict, Sequence[float], :class:`ExprRef`, :class:`Vector2number`
        The projection's center, a two-element array of longitude and latitude in degrees.

        **Default value:** ``[0, 0]``
    clipAngle : dict, float, :class:`ExprRef`
        The projection's clipping circle radius to the specified angle in degrees. If
        ``null``, switches to `antimeridian <http://bl.ocks.org/mbostock/3788999>`__ cutting
        rather than small-circle clipping.
    clipExtent : dict, :class:`ExprRef`, :class:`Vector2Vector2number`, Sequence[Sequence[float], :class:`Vector2number`]
        The projection's viewport clip extent to the specified bounds in pixels. The extent
        bounds are specified as an array ``[[x0, y0], [x1, y1]]``, where ``x0`` is the
        left-side of the viewport, ``y0`` is the top, ``x1`` is the right and ``y1`` is the
        bottom. If ``null``, no viewport clipping is performed.
    coefficient : dict, float, :class:`ExprRef`
        The coefficient parameter for the ``hammer`` projection.

        **Default value:** ``2``
    distance : dict, float, :class:`ExprRef`
        For the ``satellite`` projection, the distance from the center of the sphere to the
        point of view, as a proportion of the sphere’s radius. The recommended maximum clip
        angle for a given ``distance`` is acos(1 / distance) converted to degrees. If tilt
        is also applied, then more conservative clipping may be necessary.

        **Default value:** ``2.0``
    extent : dict, :class:`ExprRef`, :class:`Vector2Vector2number`, Sequence[Sequence[float], :class:`Vector2number`]

    fit : dict, :class:`Fit`, :class:`ExprRef`, :class:`GeoJsonFeature`, :class:`GeoJsonFeatureCollection`, Sequence[dict, :class:`GeoJsonFeature`], Sequence[dict, :class:`Fit`, :class:`GeoJsonFeature`, :class:`GeoJsonFeatureCollection`, Sequence[dict, :class:`GeoJsonFeature`]]

    fraction : dict, float, :class:`ExprRef`
        The fraction parameter for the ``bottomley`` projection.

        **Default value:** ``0.5``, corresponding to a sin(ψ) where ψ = π/6.
    lobes : dict, float, :class:`ExprRef`
        The number of lobes in projections that support multi-lobe views: ``berghaus``,
        ``gingery``, or ``healpix``. The default value varies based on the projection type.
    parallel : dict, float, :class:`ExprRef`
        The parallel parameter for projections that support it: ``armadillo``, ``bonne``,
        ``craig``, ``cylindricalEqualArea``, ``cylindricalStereographic``,
        ``hammerRetroazimuthal``, ``loximuthal``, or ``rectangularPolyconic``. The default
        value varies based on the projection type.
    parallels : dict, Sequence[float], :class:`ExprRef`
        For conic projections, the `two standard parallels
        <https://en.wikipedia.org/wiki/Map_projection#Conic>`__ that define the map layout.
        The default depends on the specific conic projection used.
    pointRadius : dict, float, :class:`ExprRef`
        The default radius (in pixels) to use when drawing GeoJSON ``Point`` and
        ``MultiPoint`` geometries. This parameter sets a constant default value. To modify
        the point radius in response to data, see the corresponding parameter of the GeoPath
        and GeoShape transforms.

        **Default value:** ``4.5``
    precision : dict, float, :class:`ExprRef`
        The threshold for the projection's `adaptive resampling
        <http://bl.ocks.org/mbostock/3795544>`__ to the specified value in pixels. This
        value corresponds to the `Douglas–Peucker distance
        <http://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm>`__.
        If precision is not specified, returns the projection's current resampling precision
        which defaults to ``√0.5 ≅ 0.70710…``.
    radius : dict, float, :class:`ExprRef`
        The radius parameter for the ``airy`` or ``gingery`` projection. The default value
        varies based on the projection type.
    ratio : dict, float, :class:`ExprRef`
        The ratio parameter for the ``hill``, ``hufnagel``, or ``wagner`` projections. The
        default value varies based on the projection type.
    reflectX : bool, dict, :class:`ExprRef`
        Sets whether or not the x-dimension is reflected (negated) in the output.
    reflectY : bool, dict, :class:`ExprRef`
        Sets whether or not the y-dimension is reflected (negated) in the output.
    rotate : dict, Sequence[float], :class:`ExprRef`, :class:`Vector2number`, :class:`Vector3number`
        The projection's three-axis rotation to the specified angles, which must be a two-
        or three-element array of numbers [ ``lambda``, ``phi``, ``gamma`` ] specifying the
        rotation angles in degrees about each spherical axis. (These correspond to yaw,
        pitch and roll.)

        **Default value:** ``[0, 0, 0]``
    scale : dict, float, :class:`ExprRef`
        The projection’s scale (zoom) factor, overriding automatic fitting. The default
        scale is projection-specific. The scale factor corresponds linearly to the distance
        between projected points; however, scale factor values are not equivalent across
        projections.
    size : dict, Sequence[float], :class:`ExprRef`, :class:`Vector2number`
        Used in conjunction with fit, provides the width and height in pixels of the area to
        which the projection should be automatically fit.
    spacing : dict, float, :class:`ExprRef`
        The spacing parameter for the ``lagrange`` projection.

        **Default value:** ``0.5``
    tilt : dict, float, :class:`ExprRef`
        The tilt angle (in degrees) for the ``satellite`` projection.

        **Default value:** ``0``.
    translate : dict, Sequence[float], :class:`ExprRef`, :class:`Vector2number`
        The projection’s translation offset as a two-element array ``[tx, ty]``.
    type : dict, :class:`ExprRef`, :class:`ProjectionType`, Literal['albers', 'albersUsa', 'azimuthalEqualArea', 'azimuthalEquidistant', 'conicConformal', 'conicEqualArea', 'conicEquidistant', 'equalEarth', 'equirectangular', 'gnomonic', 'identity', 'mercator', 'naturalEarth1', 'orthographic', 'stereographic', 'transverseMercator']
        The cartographic projection to use. This value is case-insensitive, for example
        ``"albers"`` and ``"Albers"`` indicate the same projection type. You can find all
        valid projection types `in the documentation
        <https://vega.github.io/vega-lite/docs/projection.html#projection-types>`__.

        **Default value:** ``equalEarth``
    """
    _schema = {'$ref': '#/definitions/ProjectionConfig'}

    def __init__(self, center: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, clipAngle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, clipExtent: Union[dict, '_Parameter', 'SchemaBase', Sequence[Union['SchemaBase', Sequence[float]]], UndefinedType]=Undefined, coefficient: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, distance: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, extent: Union[dict, '_Parameter', 'SchemaBase', Sequence[Union['SchemaBase', Sequence[float]]], UndefinedType]=Undefined, fit: Union[dict, '_Parameter', 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], Sequence[Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']]]], UndefinedType]=Undefined, fraction: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, lobes: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, parallel: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, parallels: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, pointRadius: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, precision: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, radius: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, ratio: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, reflectX: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, reflectY: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, rotate: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, scale: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, size: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, spacing: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tilt: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, translate: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, type: Union[dict, '_Parameter', 'SchemaBase', Literal['albers', 'albersUsa', 'azimuthalEqualArea', 'azimuthalEquidistant', 'conicConformal', 'conicEqualArea', 'conicEquidistant', 'equalEarth', 'equirectangular', 'gnomonic', 'identity', 'mercator', 'naturalEarth1', 'orthographic', 'stereographic', 'transverseMercator'], UndefinedType]=Undefined, **kwds):
        super(ProjectionConfig, self).__init__(center=center, clipAngle=clipAngle, clipExtent=clipExtent, coefficient=coefficient, distance=distance, extent=extent, fit=fit, fraction=fraction, lobes=lobes, parallel=parallel, parallels=parallels, pointRadius=pointRadius, precision=precision, radius=radius, ratio=ratio, reflectX=reflectX, reflectY=reflectY, rotate=rotate, scale=scale, size=size, spacing=spacing, tilt=tilt, translate=translate, type=type, **kwds)
import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
class TopLevelMixin(mixins.ConfigMethodMixin):
    """Mixin for top-level chart objects such as Chart, LayeredChart, etc."""
    _class_is_valid_at_instantiation: bool = False

    def to_dict(self, validate: bool=True, *, format: str='vega-lite', ignore: Optional[List[str]]=None, context: Optional[TypingDict[str, Any]]=None) -> dict:
        """Convert the chart to a dictionary suitable for JSON export

        Parameters
        ----------
        validate : bool, optional
            If True (default), then validate the output dictionary
            against the schema.
        format : str, optional
            Chart specification format, one of "vega-lite" (default) or "vega"
        ignore : list[str], optional
            A list of keys to ignore. It is usually not needed
            to specify this argument as a user.
        context : dict[str, Any], optional
            A context dictionary. It is usually not needed
            to specify this argument as a user.

        Notes
        -----
        Technical: The ignore parameter will *not* be passed to child to_dict
        function calls.

        Returns
        -------
        dict
            The dictionary representation of this chart

        Raises
        ------
        SchemaValidationError
            if validate=True and the dict does not conform to the schema
        """
        if format not in ('vega-lite', 'vega'):
            raise ValueError(f'The format argument must be either "vega-lite" or "vega". Received {repr(format)}')
        context = context.copy() if context else {}
        context.setdefault('datasets', {})
        is_top_level = context.get('top_level', True)
        copy = self.copy(deep=False)
        original_data = getattr(copy, 'data', Undefined)
        copy.data = _prepare_data(original_data, context)
        if original_data is not Undefined:
            context['data'] = original_data
        context['top_level'] = False
        vegalite_spec = super(TopLevelMixin, copy).to_dict(validate=validate, ignore=ignore, context=dict(context, pre_transform=False))
        if is_top_level:
            if '$schema' not in vegalite_spec:
                vegalite_spec['$schema'] = SCHEMA_URL
            the_theme = themes.get()
            assert the_theme is not None
            vegalite_spec = utils.update_nested(the_theme(), vegalite_spec, copy=True)
            if context['datasets']:
                vegalite_spec.setdefault('datasets', {}).update(context['datasets'])
        if context.get('pre_transform', True) and _using_vegafusion():
            if format == 'vega-lite':
                raise ValueError('When the "vegafusion" data transformer is enabled, the \nto_dict() and to_json() chart methods must be called with format="vega". \nFor example: \n    >>> chart.to_dict(format="vega")\n    >>> chart.to_json(format="vega")')
            else:
                return _compile_with_vegafusion(vegalite_spec)
        elif format == 'vega':
            plugin = vegalite_compilers.get()
            if plugin is None:
                raise ValueError('No active vega-lite compiler plugin found')
            return plugin(vegalite_spec)
        else:
            return vegalite_spec

    def to_json(self, validate: bool=True, indent: Optional[Union[int, str]]=2, sort_keys: bool=True, *, format: str='vega-lite', ignore: Optional[List[str]]=None, context: Optional[TypingDict[str, Any]]=None, **kwargs) -> str:
        """Convert a chart to a JSON string

        Parameters
        ----------
        validate : bool, optional
            If True (default), then validate the output dictionary
            against the schema.
        indent : int, optional
            The number of spaces of indentation to use. The default is 2.
        sort_keys : bool, optional
            If True (default), sort keys in the output.
        format : str, optional
            The chart specification format. One of "vega-lite" (default) or "vega".
            The "vega" format relies on the active Vega-Lite compiler plugin, which
            by default requires the vl-convert-python package.
        ignore : list[str], optional
            A list of keys to ignore. It is usually not needed
            to specify this argument as a user.
        context : dict[str, Any], optional
            A context dictionary. It is usually not needed
            to specify this argument as a user.
        **kwargs
            Additional keyword arguments are passed to ``json.dumps()``
        """
        if ignore is None:
            ignore = []
        if context is None:
            context = {}
        spec = self.to_dict(validate=validate, format=format, ignore=ignore, context=context)
        return json.dumps(spec, indent=indent, sort_keys=sort_keys, **kwargs)

    def to_html(self, base_url: str='https://cdn.jsdelivr.net/npm', output_div: str='vis', embed_options: Optional[dict]=None, json_kwds: Optional[dict]=None, fullhtml: bool=True, requirejs: bool=False, inline: bool=False, **kwargs) -> str:
        """Embed a Vega/Vega-Lite spec into an HTML page

        Parameters
        ----------
        base_url : string (optional)
            The base url from which to load the javascript libraries.
        output_div : string (optional)
            The id of the div element where the plot will be shown.
        embed_options : dict (optional)
            Dictionary of options to pass to the vega-embed script. Default
            entry is {'mode': mode}.
        json_kwds : dict (optional)
            Dictionary of keywords to pass to json.dumps().
        fullhtml : boolean (optional)
            If True (default) then return a full html page. If False, then return
            an HTML snippet that can be embedded into an HTML page.
        requirejs : boolean (optional)
            If False (default) then load libraries from base_url using <script>
            tags. If True, then load libraries using requirejs
        inline: bool (optional)
            If False (default), the required JavaScript libraries are loaded
            from a CDN location in the resulting html file.
            If True, the required JavaScript libraries are inlined into the resulting
            html file so that it will work without an internet connection.
            The vl-convert-python package is required if True.
        **kwargs :
            additional kwargs passed to spec_to_html.
        Returns
        -------
        output : string
            an HTML string for rendering the chart.
        """
        if inline:
            kwargs['template'] = 'inline'
        return utils.spec_to_html(self.to_dict(), mode='vega-lite', vegalite_version=VEGALITE_VERSION, vegaembed_version=VEGAEMBED_VERSION, vega_version=VEGA_VERSION, base_url=base_url, output_div=output_div, embed_options=embed_options, json_kwds=json_kwds, fullhtml=fullhtml, requirejs=requirejs, **kwargs)

    def to_url(self, *, fullscreen: bool=False) -> str:
        """Convert a chart to a URL that opens the chart specification in the Vega chart editor
        The chart specification (including any inline data) is encoded in the URL.

        This method requires that the vl-convert-python package is installed.

        Parameters
        ----------
        fullscreen : bool
            If True, editor will open chart in fullscreen mode. Default False
        """
        from ...utils._importers import import_vl_convert
        vlc = import_vl_convert()
        if _using_vegafusion():
            return vlc.vega_to_url(self.to_dict(format='vega'), fullscreen=fullscreen)
        else:
            return vlc.vegalite_to_url(self.to_dict(), fullscreen=fullscreen)

    def open_editor(self, *, fullscreen: bool=False) -> None:
        """Opens the chart specification in the Vega chart editor using the default browser.

        Parameters
        ----------
        fullscreen : bool
            If True, editor will open chart in fullscreen mode. Default False
        """
        import webbrowser
        webbrowser.open(self.to_url(fullscreen=fullscreen))

    def save(self, fp: Union[str, IO], format: Optional[Literal['json', 'html', 'png', 'svg', 'pdf']]=None, override_data_transformer: bool=True, scale_factor: float=1.0, mode: Optional[str]=None, vegalite_version: str=VEGALITE_VERSION, vega_version: str=VEGA_VERSION, vegaembed_version: str=VEGAEMBED_VERSION, embed_options: Optional[dict]=None, json_kwds: Optional[dict]=None, webdriver: Optional[str]=None, engine: Optional[str]=None, inline=False, **kwargs) -> None:
        """Save a chart to file in a variety of formats

        Supported formats are json, html, png, svg, pdf; the last three require
        the altair_saver package to be installed.

        Parameters
        ----------
        fp : string filename or file-like object
            file in which to write the chart.
        format : string (optional)
            the format to write: one of ['json', 'html', 'png', 'svg', 'pdf'].
            If not specified, the format will be determined from the filename.
        override_data_transformer : `boolean` (optional)
            If True (default), then the save action will be done with
            the MaxRowsError disabled. If False, then do not change the data
            transformer.
        scale_factor : float (optional)
            scale_factor to use to change size/resolution of png or svg output
        mode : string (optional)
            Must be 'vega-lite'. If not specified, then infer the mode from
            the '$schema' property of the spec, or the ``opt`` dictionary.
            If it's not specified in either of those places, then use 'vega-lite'.
        vegalite_version : string (optional)
            For html output, the version of vegalite.js to use
        vega_version : string (optional)
            For html output, the version of vega.js to use
        vegaembed_version : string (optional)
            For html output, the version of vegaembed.js to use
        embed_options : dict (optional)
            The vegaEmbed options dictionary. Default is {}
            (See https://github.com/vega/vega-embed for details)
        json_kwds : dict (optional)
            Additional keyword arguments are passed to the output method
            associated with the specified format.
        webdriver : string {'chrome' | 'firefox'} (optional)
            Webdriver to use for png, svg, or pdf output when using altair_saver engine
        engine: string {'vl-convert', 'altair_saver'}
            the conversion engine to use for 'png', 'svg', and 'pdf' formats
        inline: bool (optional)
            If False (default), the required JavaScript libraries are loaded
            from a CDN location in the resulting html file.
            If True, the required JavaScript libraries are inlined into the resulting
            html file so that it will work without an internet connection.
            The vl-convert-python package is required if True.
        **kwargs :
            additional kwargs passed to spec_to_mimebundle.
        """
        from ...utils.save import save
        kwds = dict(chart=self, fp=fp, format=format, scale_factor=scale_factor, mode=mode, vegalite_version=vegalite_version, vega_version=vega_version, vegaembed_version=vegaembed_version, embed_options=embed_options, json_kwds=json_kwds, webdriver=webdriver, engine=engine, inline=inline, **kwargs)
        if override_data_transformer:
            with data_transformers.disable_max_rows():
                save(**kwds)
        else:
            save(**kwds)
        return

    def __repr__(self) -> str:
        return 'alt.{}(...)'.format(self.__class__.__name__)

    def __add__(self, other) -> 'LayerChart':
        if not isinstance(other, TopLevelMixin):
            raise ValueError('Only Chart objects can be layered.')
        return layer(self, other)

    def __and__(self, other) -> 'VConcatChart':
        if not isinstance(other, TopLevelMixin):
            raise ValueError('Only Chart objects can be concatenated.')
        return vconcat(self, other)

    def __or__(self, other) -> 'HConcatChart':
        if not isinstance(other, TopLevelMixin):
            raise ValueError('Only Chart objects can be concatenated.')
        return hconcat(self, other)

    def repeat(self, repeat: Union[List[str], UndefinedType]=Undefined, row: Union[List[str], UndefinedType]=Undefined, column: Union[List[str], UndefinedType]=Undefined, layer: Union[List[str], UndefinedType]=Undefined, columns: Union[int, UndefinedType]=Undefined, **kwargs) -> 'RepeatChart':
        """Return a RepeatChart built from the chart

        Fields within the chart can be set to correspond to the row or
        column using `alt.repeat('row')` and `alt.repeat('column')`.

        Parameters
        ----------
        repeat : list
            a list of data column names to be repeated. This cannot be
            used along with the ``row``, ``column`` or ``layer`` argument.
        row : list
            a list of data column names to be mapped to the row facet
        column : list
            a list of data column names to be mapped to the column facet
        layer : list
            a list of data column names to be layered. This cannot be
            used along with the ``row``, ``column`` or ``repeat`` argument.
        columns : int
            the maximum number of columns before wrapping. Only referenced
            if ``repeat`` is specified.
        **kwargs :
            additional keywords passed to RepeatChart.

        Returns
        -------
        chart : RepeatChart
            a repeated chart.
        """
        repeat_specified = repeat is not Undefined
        rowcol_specified = row is not Undefined or column is not Undefined
        layer_specified = layer is not Undefined
        if repeat_specified and rowcol_specified:
            raise ValueError('repeat argument cannot be combined with row/column argument.')
        elif repeat_specified and layer_specified:
            raise ValueError('repeat argument cannot be combined with layer argument.')
        repeat_arg: Union[List[str], core.LayerRepeatMapping, core.RepeatMapping]
        if repeat_specified:
            assert not isinstance(repeat, UndefinedType)
            repeat_arg = repeat
        elif layer_specified:
            repeat_arg = core.LayerRepeatMapping(layer=layer, row=row, column=column)
        else:
            repeat_arg = core.RepeatMapping(row=row, column=column)
        return RepeatChart(spec=self, repeat=repeat_arg, columns=columns, **kwargs)

    def properties(self, **kwargs) -> Self:
        """Set top-level properties of the Chart.

        Argument names and types are the same as class initialization.
        """
        copy = self.copy(deep=False)
        for key, val in kwargs.items():
            if key == 'selection' and isinstance(val, Parameter):
                setattr(copy, key, {val.name: val.selection})
            else:
                if key != 'data':
                    self.validate_property(key, val)
                setattr(copy, key, val)
        return copy

    def project(self, type: Union[str, core.ProjectionType, core.ExprRef, Parameter, UndefinedType]=Undefined, center: Union[List[float], core.Vector2number, core.ExprRef, Parameter, UndefinedType]=Undefined, clipAngle: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, clipExtent: Union[List[List[float]], core.Vector2Vector2number, core.ExprRef, Parameter, UndefinedType]=Undefined, coefficient: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, distance: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, fraction: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, lobes: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, parallel: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, precision: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, radius: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, ratio: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, reflectX: Union[bool, core.ExprRef, Parameter, UndefinedType]=Undefined, reflectY: Union[bool, core.ExprRef, Parameter, UndefinedType]=Undefined, rotate: Union[List[float], core.Vector2number, core.Vector3number, core.ExprRef, Parameter, UndefinedType]=Undefined, scale: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, spacing: Union[float, core.Vector2number, core.ExprRef, Parameter, UndefinedType]=Undefined, tilt: Union[float, core.ExprRef, Parameter, UndefinedType]=Undefined, translate: Union[List[float], core.Vector2number, core.ExprRef, Parameter, UndefinedType]=Undefined, **kwds) -> Self:
        """Add a geographic projection to the chart.

        This is generally used either with ``mark_geoshape`` or with the
        ``latitude``/``longitude`` encodings.

        Available projection types are
        ['albers', 'albersUsa', 'azimuthalEqualArea', 'azimuthalEquidistant',
        'conicConformal', 'conicEqualArea', 'conicEquidistant', 'equalEarth', 'equirectangular',
        'gnomonic', 'identity', 'mercator', 'orthographic', 'stereographic', 'transverseMercator']

        Parameters
        ----------
        type : str
            The cartographic projection to use. This value is case-insensitive, for example
            `"albers"` and `"Albers"` indicate the same projection type. You can find all valid
            projection types [in the
            documentation](https://vega.github.io/vega-lite/docs/projection.html#projection-types).

            **Default value:** `equalEarth`
        center : List(float)
            Sets the projection’s center to the specified center, a two-element array of
            longitude and latitude in degrees.

            **Default value:** `[0, 0]`
        clipAngle : float
            Sets the projection’s clipping circle radius to the specified angle in degrees. If
            `null`, switches to [antimeridian](http://bl.ocks.org/mbostock/3788999) cutting
            rather than small-circle clipping.
        clipExtent : List(List(float))
            Sets the projection’s viewport clip extent to the specified bounds in pixels. The
            extent bounds are specified as an array `[[x0, y0], [x1, y1]]`, where `x0` is the
            left-side of the viewport, `y0` is the top, `x1` is the right and `y1` is the
            bottom. If `null`, no viewport clipping is performed.
        coefficient : float
            The coefficient parameter for the ``hammer`` projection.

            **Default value:** ``2``
        distance : float
            For the ``satellite`` projection, the distance from the center of the sphere to the
            point of view, as a proportion of the sphere’s radius. The recommended maximum clip
            angle for a given ``distance`` is acos(1 / distance) converted to degrees. If tilt
            is also applied, then more conservative clipping may be necessary.

            **Default value:** ``2.0``
        fraction : float
            The fraction parameter for the ``bottomley`` projection.

            **Default value:** ``0.5``, corresponding to a sin(ψ) where ψ = π/6.
        lobes : float
            The number of lobes in projections that support multi-lobe views: ``berghaus``,
            ``gingery``, or ``healpix``. The default value varies based on the projection type.
        parallel : float
            For conic projections, the `two standard parallels
            <https://en.wikipedia.org/wiki/Map_projection#Conic>`__ that define the map layout.
            The default depends on the specific conic projection used.
        precision : float
            Sets the threshold for the projection’s [adaptive
            resampling](http://bl.ocks.org/mbostock/3795544) to the specified value in pixels.
            This value corresponds to the [Douglas–Peucker
            distance](http://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).
             If precision is not specified, returns the projection’s current resampling
            precision which defaults to `√0.5 ≅ 0.70710…`.
        radius : float
            The radius parameter for the ``airy`` or ``gingery`` projection. The default value
            varies based on the projection type.
        ratio : float
            The ratio parameter for the ``hill``, ``hufnagel``, or ``wagner`` projections. The
            default value varies based on the projection type.
        reflectX : boolean
            Sets whether or not the x-dimension is reflected (negated) in the output.
        reflectY : boolean
            Sets whether or not the y-dimension is reflected (negated) in the output.
        rotate : List(float)
            Sets the projection’s three-axis rotation to the specified angles, which must be a
            two- or three-element array of numbers [`lambda`, `phi`, `gamma`] specifying the
            rotation angles in degrees about each spherical axis. (These correspond to yaw,
            pitch and roll.)

            **Default value:** `[0, 0, 0]`
        scale : float
            The projection’s scale (zoom) factor, overriding automatic fitting. The default
            scale is projection-specific. The scale factor corresponds linearly to the distance
            between projected points; however, scale factor values are not equivalent across
            projections.
        spacing : float
            The spacing parameter for the ``lagrange`` projection.

            **Default value:** ``0.5``
        tilt : float
            The tilt angle (in degrees) for the ``satellite`` projection.

            **Default value:** ``0``.
        translate : List(float)
            The projection’s translation offset as a two-element array ``[tx, ty]``,
            overriding automatic fitting.

        """
        projection = core.Projection(center=center, clipAngle=clipAngle, clipExtent=clipExtent, coefficient=coefficient, distance=distance, fraction=fraction, lobes=lobes, parallel=parallel, precision=precision, radius=radius, ratio=ratio, reflectX=reflectX, reflectY=reflectY, rotate=rotate, scale=scale, spacing=spacing, tilt=tilt, translate=translate, type=type, **kwds)
        return self.properties(projection=projection)

    def _add_transform(self, *transforms: core.Transform) -> Self:
        """Copy the chart and add specified transforms to chart.transform"""
        copy = self.copy(deep=['transform'])
        if copy.transform is Undefined:
            copy.transform = []
        copy.transform.extend(transforms)
        return copy

    def transform_aggregate(self, aggregate: Union[List[core.AggregatedFieldDef], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, **kwds: Union[TypingDict[str, Any], str]) -> Self:
        """
        Add an :class:`AggregateTransform` to the schema.

        Parameters
        ----------
        aggregate : List(:class:`AggregatedFieldDef`)
            Array of objects that define fields to aggregate.
        groupby : List(string)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        **kwds : Union[TypingDict[str, Any], str]
            additional keywords are converted to aggregates using standard
            shorthand parsing.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        The aggregate transform allows you to specify transforms directly using
        the same shorthand syntax as used in encodings:

        >>> import altair as alt
        >>> chart1 = alt.Chart().transform_aggregate(
        ...     mean_acc='mean(Acceleration)',
        ...     groupby=['Origin']
        ... )
        >>> print(chart1.transform[0].to_json())  # doctest: +NORMALIZE_WHITESPACE
        {
          "aggregate": [
            {
              "as": "mean_acc",
              "field": "Acceleration",
              "op": "mean"
            }
          ],
          "groupby": [
            "Origin"
          ]
        }

        It also supports including AggregatedFieldDef instances or dicts directly,
        so you can create the above transform like this:

        >>> chart2 = alt.Chart().transform_aggregate(
        ...     [alt.AggregatedFieldDef(field='Acceleration', op='mean',
        ...                             **{'as': 'mean_acc'})],
        ...     groupby=['Origin']
        ... )
        >>> chart2.transform == chart1.transform
        True

        See Also
        --------
        alt.AggregateTransform : underlying transform object

        """
        if aggregate is Undefined:
            aggregate = []
        for key, val in kwds.items():
            parsed = utils.parse_shorthand(val)
            dct = {'as': key, 'field': parsed.get('field', Undefined), 'op': parsed.get('aggregate', Undefined)}
            assert not isinstance(aggregate, UndefinedType)
            aggregate.append(core.AggregatedFieldDef(**dct))
        return self._add_transform(core.AggregateTransform(aggregate=aggregate, groupby=groupby))

    def transform_bin(self, as_: Union[str, core.FieldName, List[Union[str, core.FieldName]], UndefinedType]=Undefined, field: Union[str, core.FieldName, UndefinedType]=Undefined, bin: Union[Literal[True], core.BinParams]=True, **kwargs) -> Self:
        """
        Add a :class:`BinTransform` to the schema.

        Parameters
        ----------
        as_ : anyOf(string, List(string))
            The output fields at which to write the start and end bin values.
        bin : anyOf(boolean, :class:`BinParams`)
            An object indicating bin properties, or simply ``true`` for using default bin
            parameters.
        field : string
            The data field to bin.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> chart = alt.Chart().transform_bin("x_binned", "x")
        >>> chart.transform[0]
        BinTransform({
          as: 'x_binned',
          bin: True,
          field: 'x'
        })

        >>> chart = alt.Chart().transform_bin("x_binned", "x",
        ...                                   bin=alt.Bin(maxbins=10))
        >>> chart.transform[0]
        BinTransform({
          as: 'x_binned',
          bin: BinParams({
            maxbins: 10
          }),
          field: 'x'
        })

        See Also
        --------
        alt.BinTransform : underlying transform object

        """
        if as_ is not Undefined:
            if 'as' in kwargs:
                raise ValueError("transform_bin: both 'as_' and 'as' passed as arguments.")
            kwargs['as'] = as_
        kwargs['bin'] = bin
        kwargs['field'] = field
        return self._add_transform(core.BinTransform(**kwargs))

    def transform_calculate(self, as_: Union[str, core.FieldName, UndefinedType]=Undefined, calculate: Union[str, core.Expr, _expr_core.Expression, UndefinedType]=Undefined, **kwargs: Union[str, core.Expr, _expr_core.Expression]) -> Self:
        """
        Add a :class:`CalculateTransform` to the schema.

        Parameters
        ----------
        as_ : string
            The field for storing the computed formula value.
        calculate : string or alt.expr.Expression
            An `expression <https://vega.github.io/vega-lite/docs/types.html#expression>`__
            string. Use the variable ``datum`` to refer to the current data object.
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> from altair import datum, expr

        >>> chart = alt.Chart().transform_calculate(y = 2 * expr.sin(datum.x))
        >>> chart.transform[0]
        CalculateTransform({
          as: 'y',
          calculate: (2 * sin(datum.x))
        })

        It's also possible to pass the ``CalculateTransform`` arguments directly:

        >>> kwds = {'as_': 'y', 'calculate': '2 * sin(datum.x)'}
        >>> chart = alt.Chart().transform_calculate(**kwds)
        >>> chart.transform[0]
        CalculateTransform({
          as: 'y',
          calculate: '2 * sin(datum.x)'
        })

        As the first form is easier to write and understand, that is the
        recommended method.

        See Also
        --------
        alt.CalculateTransform : underlying transform object
        """
        if as_ is Undefined:
            as_ = kwargs.pop('as', Undefined)
        elif 'as' in kwargs:
            raise ValueError("transform_calculate: both 'as_' and 'as' passed as arguments.")
        if as_ is not Undefined or calculate is not Undefined:
            dct = {'as': as_, 'calculate': calculate}
            self = self._add_transform(core.CalculateTransform(**dct))
        for as_, calculate in kwargs.items():
            dct = {'as': as_, 'calculate': calculate}
            self = self._add_transform(core.CalculateTransform(**dct))
        return self

    def transform_density(self, density: Union[str, core.FieldName], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, bandwidth: Union[float, UndefinedType]=Undefined, counts: Union[bool, UndefinedType]=Undefined, cumulative: Union[bool, UndefinedType]=Undefined, extent: Union[List[float], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, maxsteps: Union[int, UndefinedType]=Undefined, minsteps: Union[int, UndefinedType]=Undefined, steps: Union[int, UndefinedType]=Undefined) -> Self:
        """Add a :class:`DensityTransform` to the spec.

        Parameters
        ----------
        density : str
            The data field for which to perform density estimation.
        as_ : [str, str]
            The output fields for the sample value and corresponding density estimate.
            **Default value:** ``["value", "density"]``
        bandwidth : float
            The bandwidth (standard deviation) of the Gaussian kernel. If unspecified or set to
            zero, the bandwidth value is automatically estimated from the input data using
            Scott’s rule.
        counts : boolean
            A boolean flag indicating if the output values should be probability estimates
            (false) or smoothed counts (true).
            **Default value:** ``false``
        cumulative : boolean
            A boolean flag indicating whether to produce density estimates (false) or cumulative
            density estimates (true).
            **Default value:** ``false``
        extent : List([float, float])
            A [min, max] domain from which to sample the distribution. If unspecified, the
            extent will be determined by the observed minimum and maximum values of the density
            value field.
        groupby : List(str)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        maxsteps : int
            The maximum number of samples to take along the extent domain for plotting the
            density. **Default value:** ``200``
        minsteps : int
            The minimum number of samples to take along the extent domain for plotting the
            density. **Default value:** ``25``
        steps : int
            The exact number of samples to take along the extent domain for plotting the
            density. If specified, overrides both minsteps and maxsteps to set an exact number
            of uniform samples. Potentially useful in conjunction with a fixed extent to ensure
            consistent sample points for stacked densities.
        """
        return self._add_transform(core.DensityTransform(density=density, bandwidth=bandwidth, counts=counts, cumulative=cumulative, extent=extent, groupby=groupby, maxsteps=maxsteps, minsteps=minsteps, steps=steps, **{'as': as_}))

    def transform_impute(self, impute: Union[str, core.FieldName], key: Union[str, core.FieldName], frame: Union[List[Optional[int]], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, keyvals: Union[List[Any], core.ImputeSequence, UndefinedType]=Undefined, method: Union[Literal['value', 'mean', 'median', 'max', 'min'], core.ImputeMethod, UndefinedType]=Undefined, value=Undefined) -> Self:
        """
        Add an :class:`ImputeTransform` to the schema.

        Parameters
        ----------
        impute : string
            The data field for which the missing values should be imputed.
        key : string
            A key field that uniquely identifies data objects within a group.
            Missing key values (those occurring in the data but not in the current group) will
            be imputed.
        frame : List(anyOf(None, int))
            A frame specification as a two-element array used to control the window over which
            the specified method is applied. The array entries should either be a number
            indicating the offset from the current data object, or null to indicate unbounded
            rows preceding or following the current data object.  For example, the value ``[-5,
            5]`` indicates that the window should include five objects preceding and five
            objects following the current object.
            **Default value:** :  ``[null, null]`` indicating that the window includes all
            objects.
        groupby : List(string)
            An optional array of fields by which to group the values.
            Imputation will then be performed on a per-group basis.
        keyvals : anyOf(List(Mapping(required=[])), :class:`ImputeSequence`)
            Defines the key values that should be considered for imputation.
            An array of key values or an object defining a `number sequence
            <https://vega.github.io/vega-lite/docs/impute.html#sequence-def>`__.
            If provided, this will be used in addition to the key values observed within the
            input data.  If not provided, the values will be derived from all unique values of
            the ``key`` field. For ``impute`` in ``encoding``, the key field is the x-field if
            the y-field is imputed, or vice versa.
            If there is no impute grouping, this property *must* be specified.
        method : :class:`ImputeMethod`
            The imputation method to use for the field value of imputed data objects.
            One of ``value``, ``mean``, ``median``, ``max`` or ``min``.
            **Default value:**  ``"value"``
        value : Mapping(required=[])
            The field value to use when the imputation ``method`` is ``"value"``.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.ImputeTransform : underlying transform object
        """
        return self._add_transform(core.ImputeTransform(impute=impute, key=key, frame=frame, groupby=groupby, keyvals=keyvals, method=method, value=value))

    def transform_joinaggregate(self, joinaggregate: Union[List[core.JoinAggregateFieldDef], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, **kwargs: str) -> Self:
        """
        Add a :class:`JoinAggregateTransform` to the schema.

        Parameters
        ----------
        joinaggregate : List(:class:`JoinAggregateFieldDef`)
            The definition of the fields in the join aggregate, and what calculations to use.
        groupby : List(string)
            The data fields for partitioning the data objects into separate groups. If
            unspecified, all data points will be in a single group.
        **kwargs
            joinaggregates can also be passed by keyword argument; see Examples.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> chart = alt.Chart().transform_joinaggregate(x='sum(y)')
        >>> chart.transform[0]
        JoinAggregateTransform({
          joinaggregate: [JoinAggregateFieldDef({
            as: 'x',
            field: 'y',
            op: 'sum'
          })]
        })

        See Also
        --------
        alt.JoinAggregateTransform : underlying transform object
        """
        if joinaggregate is Undefined:
            joinaggregate = []
        for key, val in kwargs.items():
            parsed = utils.parse_shorthand(val)
            dct = {'as': key, 'field': parsed.get('field', Undefined), 'op': parsed.get('aggregate', Undefined)}
            assert not isinstance(joinaggregate, UndefinedType)
            joinaggregate.append(core.JoinAggregateFieldDef(**dct))
        return self._add_transform(core.JoinAggregateTransform(joinaggregate=joinaggregate, groupby=groupby))

    def transform_extent(self, extent: Union[str, core.FieldName], param: Union[str, core.ParameterName]) -> Self:
        """Add a :class:`ExtentTransform` to the spec.

        Parameters
        ----------
        extent : str
            The field of which to get the extent.
        param : str
            The name of the output parameter which will be created by
            the extent transform.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining
        """
        return self._add_transform(core.ExtentTransform(extent=extent, param=param))

    def transform_filter(self, filter: Union[str, core.Expr, _expr_core.Expression, core.Predicate, Parameter, core.PredicateComposition, TypingDict[str, Union[core.Predicate, str, list, bool]]], **kwargs) -> Self:
        """
        Add a :class:`FilterTransform` to the schema.

        Parameters
        ----------
        filter : a filter expression or :class:`PredicateComposition`
            The `filter` property must be one of the predicate definitions:
            (1) a string or alt.expr expression
            (2) a range predicate
            (3) a selection predicate
            (4) a logical operand combining (1)-(3)
            (5) a Selection object

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining
        """
        if isinstance(filter, Parameter):
            new_filter: TypingDict[str, Union[bool, str]] = {'param': filter.name}
            if 'empty' in kwargs:
                new_filter['empty'] = kwargs.pop('empty')
            elif isinstance(filter.empty, bool):
                new_filter['empty'] = filter.empty
            filter = new_filter
        return self._add_transform(core.FilterTransform(filter=filter, **kwargs))

    def transform_flatten(self, flatten: List[Union[str, core.FieldName]], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined) -> Self:
        """Add a :class:`FlattenTransform` to the schema.

        Parameters
        ----------
        flatten : List(string)
            An array of one or more data fields containing arrays to flatten.
            If multiple fields are specified, their array values should have a parallel
            structure, ideally with the same length.
            If the lengths of parallel arrays do not match,
            the longest array will be used with ``null`` values added for missing entries.
        as : List(string)
            The output field names for extracted array values.
            **Default value:** The field name of the corresponding array field

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.FlattenTransform : underlying transform object
        """
        return self._add_transform(core.FlattenTransform(flatten=flatten, **{'as': as_}))

    def transform_fold(self, fold: List[Union[str, core.FieldName]], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined) -> Self:
        """Add a :class:`FoldTransform` to the spec.

        Parameters
        ----------
        fold : List(string)
            An array of data fields indicating the properties to fold.
        as : [string, string]
            The output field names for the key and value properties produced by the fold
            transform. Default: ``["key", "value"]``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        Chart.transform_pivot : pivot transform - opposite of fold.
        alt.FoldTransform : underlying transform object
        """
        return self._add_transform(core.FoldTransform(fold=fold, **{'as': as_}))

    def transform_loess(self, on: Union[str, core.FieldName], loess: Union[str, core.FieldName], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, bandwidth: Union[float, UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined) -> Self:
        """Add a :class:`LoessTransform` to the spec.

        Parameters
        ----------
        on : str
            The data field of the independent variable to use a predictor.
        loess : str
            The data field of the dependent variable to smooth.
        as_ : [str, str]
            The output field names for the smoothed points generated by the loess transform.
            **Default value:** The field names of the input x and y values.
        bandwidth : float
            A bandwidth parameter in the range ``[0, 1]`` that determines the amount of
            smoothing. **Default value:** ``0.3``
        groupby : List(str)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        Chart.transform_regression: regression transform
        alt.LoessTransform : underlying transform object
        """
        return self._add_transform(core.LoessTransform(loess=loess, on=on, bandwidth=bandwidth, groupby=groupby, **{'as': as_}))

    def transform_lookup(self, lookup: Union[str, UndefinedType]=Undefined, from_: Union[core.LookupData, core.LookupSelection, UndefinedType]=Undefined, as_: Union[Union[str, core.FieldName], List[Union[str, core.FieldName]], UndefinedType]=Undefined, default: Union[str, UndefinedType]=Undefined, **kwargs) -> Self:
        """Add a :class:`DataLookupTransform` or :class:`SelectionLookupTransform` to the chart

        Parameters
        ----------
        lookup : string
            Key in primary data source.
        from_ : anyOf(:class:`LookupData`, :class:`LookupSelection`)
            Secondary data reference.
        as_ : anyOf(string, List(string))
            The output fields on which to store the looked up data values.

            For data lookups, this property may be left blank if ``from_.fields``
            has been specified (those field names will be used); if ``from_.fields``
            has not been specified, ``as_`` must be a string.

            For selection lookups, this property is optional: if unspecified,
            looked up values will be stored under a property named for the selection;
            and if specified, it must correspond to ``from_.fields``.
        default : string
            The default value to use if lookup fails. **Default value:** ``null``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.DataLookupTransform : underlying transform object
        alt.SelectionLookupTransform : underlying transform object
        """
        if as_ is not Undefined:
            if 'as' in kwargs:
                raise ValueError("transform_lookup: both 'as_' and 'as' passed as arguments.")
            kwargs['as'] = as_
        if from_ is not Undefined:
            if 'from' in kwargs:
                raise ValueError("transform_lookup: both 'from_' and 'from' passed as arguments.")
            kwargs['from'] = from_
        kwargs['lookup'] = lookup
        kwargs['default'] = default
        return self._add_transform(core.LookupTransform(**kwargs))

    def transform_pivot(self, pivot: Union[str, core.FieldName], value: Union[str, core.FieldName], groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, limit: Union[int, UndefinedType]=Undefined, op: Union[str, core.AggregateOp, UndefinedType]=Undefined) -> Self:
        """Add a :class:`PivotTransform` to the chart.

        Parameters
        ----------
        pivot : str
            The data field to pivot on. The unique values of this field become new field names
            in the output stream.
        value : str
            The data field to populate pivoted fields. The aggregate values of this field become
            the values of the new pivoted fields.
        groupby : List(str)
            The optional data fields to group by. If not specified, a single group containing
            all data objects will be used.
        limit : int
            An optional parameter indicating the maximum number of pivoted fields to generate.
            The default ( ``0`` ) applies no limit. The pivoted ``pivot`` names are sorted in
            ascending order prior to enforcing the limit.
            **Default value:** ``0``
        op : string
            The aggregation operation to apply to grouped ``value`` field values.
            **Default value:** ``sum``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        Chart.transform_fold : fold transform - opposite of pivot.
        alt.PivotTransform : underlying transform object
        """
        return self._add_transform(core.PivotTransform(pivot=pivot, value=value, groupby=groupby, limit=limit, op=op))

    def transform_quantile(self, quantile: Union[str, core.FieldName], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, probs: Union[List[float], UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined) -> Self:
        """Add a :class:`QuantileTransform` to the chart

        Parameters
        ----------
        quantile : str
            The data field for which to perform quantile estimation.
        as : [str, str]
            The output field names for the probability and quantile values.
        groupby : List(str)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        probs : List(float)
            An array of probabilities in the range (0, 1) for which to compute quantile values.
            If not specified, the *step* parameter will be used.
        step : float
            A probability step size (default 0.01) for sampling quantile values. All values from
            one-half the step size up to 1 (exclusive) will be sampled. This parameter is only
            used if the *probs* parameter is not provided. **Default value:** ``["prob", "value"]``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.QuantileTransform : underlying transform object
        """
        return self._add_transform(core.QuantileTransform(quantile=quantile, groupby=groupby, probs=probs, step=step, **{'as': as_}))

    def transform_regression(self, on: Union[str, core.FieldName], regression: Union[str, core.FieldName], as_: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, extent: Union[List[float], UndefinedType]=Undefined, groupby: Union[List[Union[str, core.FieldName]], UndefinedType]=Undefined, method: Union[Literal['linear', 'log', 'exp', 'pow', 'quad', 'poly'], UndefinedType]=Undefined, order: Union[int, UndefinedType]=Undefined, params: Union[bool, UndefinedType]=Undefined) -> Self:
        """Add a :class:`RegressionTransform` to the chart.

        Parameters
        ----------
        on : str
            The data field of the independent variable to use a predictor.
        regression : str
            The data field of the dependent variable to predict.
        as_ : [str, str]
            The output field names for the smoothed points generated by the regression
            transform. **Default value:** The field names of the input x and y values.
        extent : [float, float]
            A [min, max] domain over the independent (x) field for the starting and ending
            points of the generated trend line.
        groupby : List(str)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        method : enum('linear', 'log', 'exp', 'pow', 'quad', 'poly')
            The functional form of the regression model. One of ``"linear"``, ``"log"``,
            ``"exp"``, ``"pow"``, ``"quad"``, or ``"poly"``.  **Default value:** ``"linear"``
        order : int
            The polynomial order (number of coefficients) for the 'poly' method.
            **Default value:** ``3``
        params : boolean
            A boolean flag indicating if the transform should return the regression model
            parameters (one object per group), rather than trend line points.
            The resulting objects include a ``coef`` array of fitted coefficient values
            (starting with the intercept term and then including terms of increasing order)
            and an ``rSquared`` value (indicating the total variance explained by the model).
            **Default value:** ``false``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        Chart.transform_loess : LOESS transform
        alt.RegressionTransform : underlying transform object
        """
        return self._add_transform(core.RegressionTransform(regression=regression, on=on, extent=extent, groupby=groupby, method=method, order=order, params=params, **{'as': as_}))

    def transform_sample(self, sample: int=1000) -> Self:
        """
        Add a :class:`SampleTransform` to the schema.

        Parameters
        ----------
        sample : int
            The maximum number of data objects to include in the sample. Default: 1000.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.SampleTransform : underlying transform object
        """
        return self._add_transform(core.SampleTransform(sample))

    def transform_stack(self, as_: Union[str, core.FieldName, List[str]], stack: Union[str, core.FieldName], groupby: List[Union[str, core.FieldName]], offset: Union[Literal['zero', 'center', 'normalize'], UndefinedType]=Undefined, sort: Union[List[core.SortField], UndefinedType]=Undefined) -> Self:
        """
        Add a :class:`StackTransform` to the schema.

        Parameters
        ----------
        as_ : anyOf(string, List(string))
            Output field names. This can be either a string or an array of strings with
            two elements denoting the name for the fields for stack start and stack end
            respectively.
            If a single string(eg."val") is provided, the end field will be "val_end".
        stack : string
            The field which is stacked.
        groupby : List(string)
            The data fields to group by.
        offset : enum('zero', 'center', 'normalize')
            Mode for stacking marks. Default: 'zero'.
        sort : List(:class:`SortField`)
            Field that determines the order of leaves in the stacked charts.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.StackTransform : underlying transform object
        """
        return self._add_transform(core.StackTransform(stack=stack, groupby=groupby, offset=offset, sort=sort, **{'as': as_}))

    def transform_timeunit(self, as_: Union[str, core.FieldName, UndefinedType]=Undefined, field: Union[str, core.FieldName, UndefinedType]=Undefined, timeUnit: Union[str, core.TimeUnit, UndefinedType]=Undefined, **kwargs: str) -> Self:
        """
        Add a :class:`TimeUnitTransform` to the schema.

        Parameters
        ----------
        as_ : string
            The output field to write the timeUnit value.
        field : string
            The data field to apply time unit.
        timeUnit : str or :class:`TimeUnit`
            The timeUnit.
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> from altair import datum, expr

        >>> chart = alt.Chart().transform_timeunit(month='month(date)')
        >>> chart.transform[0]
        TimeUnitTransform({
          as: 'month',
          field: 'date',
          timeUnit: 'month'
        })

        It's also possible to pass the ``TimeUnitTransform`` arguments directly;
        this is most useful in cases where the desired field name is not a
        valid python identifier:

        >>> kwds = {'as': 'month', 'timeUnit': 'month', 'field': 'The Month'}
        >>> chart = alt.Chart().transform_timeunit(**kwds)
        >>> chart.transform[0]
        TimeUnitTransform({
          as: 'month',
          field: 'The Month',
          timeUnit: 'month'
        })

        As the first form is easier to write and understand, that is the
        recommended method.

        See Also
        --------
        alt.TimeUnitTransform : underlying transform object

        """
        if as_ is Undefined:
            as_ = kwargs.pop('as', Undefined)
        elif 'as' in kwargs:
            raise ValueError("transform_timeunit: both 'as_' and 'as' passed as arguments.")
        if as_ is not Undefined:
            dct = {'as': as_, 'timeUnit': timeUnit, 'field': field}
            self = self._add_transform(core.TimeUnitTransform(**dct))
        for as_, shorthand in kwargs.items():
            dct = utils.parse_shorthand(shorthand, parse_timeunits=True, parse_aggregates=False, parse_types=False)
            dct.pop('type', None)
            dct['as'] = as_
            if 'timeUnit' not in dct:
                raise ValueError("'{}' must include a valid timeUnit".format(shorthand))
            self = self._add_transform(core.TimeUnitTransform(**dct))
        return self

    def transform_window(self, window: Union[List[core.WindowFieldDef], UndefinedType]=Undefined, frame: Union[List[Optional[int]], UndefinedType]=Undefined, groupby: Union[List[str], UndefinedType]=Undefined, ignorePeers: Union[bool, UndefinedType]=Undefined, sort: Union[List[Union[core.SortField, TypingDict[str, str]]], UndefinedType]=Undefined, **kwargs: str) -> Self:
        """Add a :class:`WindowTransform` to the schema

        Parameters
        ----------
        window : List(:class:`WindowFieldDef`)
            The definition of the fields in the window, and what calculations to use.
        frame : List(anyOf(None, int))
            A frame specification as a two-element array indicating how the sliding window
            should proceed. The array entries should either be a number indicating the offset
            from the current data object, or null to indicate unbounded rows preceding or
            following the current data object. The default value is ``[null, 0]``, indicating
            that the sliding window includes the current object and all preceding objects. The
            value ``[-5, 5]`` indicates that the window should include five objects preceding
            and five objects following the current object. Finally, ``[null, null]`` indicates
            that the window frame should always include all data objects. The only operators
            affected are the aggregation operations and the ``first_value``, ``last_value``, and
            ``nth_value`` window operations. The other window operations are not affected by
            this.

            **Default value:** :  ``[null, 0]`` (includes the current object and all preceding
            objects)
        groupby : List(string)
            The data fields for partitioning the data objects into separate windows. If
            unspecified, all data points will be in a single group.
        ignorePeers : boolean
            Indicates if the sliding window frame should ignore peer values. (Peer values are
            those considered identical by the sort criteria). The default is false, causing the
            window frame to expand to include all peer values. If set to true, the window frame
            will be defined by offset values only. This setting only affects those operations
            that depend on the window frame, namely aggregation operations and the first_value,
            last_value, and nth_value window operations.

            **Default value:** ``false``
        sort : List(:class:`SortField`)
            A sort field definition for sorting data objects within a window. If two data
            objects are considered equal by the comparator, they are considered “peer” values of
            equal rank. If sort is not specified, the order is undefined: data objects are
            processed in the order they are observed and none are considered peers (the
            ignorePeers parameter is ignored and treated as if set to ``true`` ).
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Examples
        --------
        A cumulative line chart

        >>> import altair as alt
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({'x': np.arange(100),
        ...                      'y': np.random.randn(100)})
        >>> chart = alt.Chart(data).mark_line().encode(
        ...     x='x:Q',
        ...     y='ycuml:Q'
        ... ).transform_window(
        ...     ycuml='sum(y)'
        ... )
        >>> chart.transform[0]
        WindowTransform({
          window: [WindowFieldDef({
            as: 'ycuml',
            field: 'y',
            op: 'sum'
          })]
        })

        """
        if kwargs:
            if window is Undefined:
                window = []
            for as_, shorthand in kwargs.items():
                kwds = {'as': as_}
                kwds.update(utils.parse_shorthand(shorthand, parse_aggregates=False, parse_window_ops=True, parse_timeunits=False, parse_types=False))
                assert not isinstance(window, UndefinedType)
                window.append(core.WindowFieldDef(**kwds))
        return self._add_transform(core.WindowTransform(window=window, frame=frame, groupby=groupby, ignorePeers=ignorePeers, sort=sort))

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return a MIME bundle for display in Jupyter frontends."""
        try:
            dct = self.to_dict(context={'pre_transform': False})
        except Exception:
            utils.display_traceback(in_ipython=True)
            return {}
        else:
            return renderers.get()(dct)

    def display(self, renderer: Union[Literal['canvas', 'svg'], UndefinedType]=Undefined, theme: Union[str, UndefinedType]=Undefined, actions: Union[bool, dict, UndefinedType]=Undefined, **kwargs) -> None:
        """Display chart in Jupyter notebook or JupyterLab

        Parameters are passed as options to vega-embed within supported frontends.
        See https://github.com/vega/vega-embed#options for details.

        Parameters
        ----------
        renderer : string ('canvas' or 'svg')
            The renderer to use
        theme : string
            The Vega theme name to use; see https://github.com/vega/vega-themes
        actions : bool or dict
            Specify whether action links ("Open In Vega Editor", etc.) are
            included in the view.
        **kwargs :
            Additional parameters are also passed to vega-embed as options.

        """
        from IPython.display import display
        if renderer is not Undefined:
            kwargs['renderer'] = renderer
        if theme is not Undefined:
            kwargs['theme'] = theme
        if actions is not Undefined:
            kwargs['actions'] = actions
        if kwargs:
            options = renderers.options.copy()
            options['embed_options'] = options.get('embed_options', {}).copy()
            options['embed_options'].update(kwargs)
            with renderers.enable(**options):
                display(self)
        else:
            display(self)

    @utils.deprecation.deprecated(message="'serve' is deprecated. Use 'show' instead.")
    def serve(self, ip='127.0.0.1', port=8888, n_retries=50, files=None, jupyter_warning=True, open_browser=True, http_server=None, **kwargs):
        """
        'serve' is deprecated. Use 'show' instead.

        Open a browser window and display a rendering of the chart

        Parameters
        ----------
        html : string
            HTML to serve
        ip : string (default = '127.0.0.1')
            ip address at which the HTML will be served.
        port : int (default = 8888)
            the port at which to serve the HTML
        n_retries : int (default = 50)
            the number of nearby ports to search if the specified port
            is already in use.
        files : dictionary (optional)
            dictionary of extra content to serve
        jupyter_warning : bool (optional)
            if True (default), then print a warning if this is used
            within the Jupyter notebook
        open_browser : bool (optional)
            if True (default), then open a web browser to the given HTML
        http_server : class (optional)
            optionally specify an HTTPServer class to use for showing the
            figure. The default is Python's basic HTTPServer.
        **kwargs :
            additional keyword arguments passed to the save() method

        """
        from ...utils.server import serve
        html = io.StringIO()
        self.save(html, format='html', **kwargs)
        html.seek(0)
        serve(html.read(), ip=ip, port=port, n_retries=n_retries, files=files, jupyter_warning=jupyter_warning, open_browser=open_browser, http_server=http_server)

    def show(self) -> None:
        """Display the chart using the active renderer"""
        if renderers.active == 'browser':
            self._repr_mimebundle_()
        else:
            from IPython.display import display
            display(self)

    @utils.use_signature(core.Resolve)
    def _set_resolve(self, **kwargs):
        """Copy the chart and update the resolve property with kwargs"""
        if not hasattr(self, 'resolve'):
            raise ValueError("{} object has no attribute 'resolve'".format(self.__class__))
        copy = self.copy(deep=['resolve'])
        if copy.resolve is Undefined:
            copy.resolve = core.Resolve()
        for key, val in kwargs.items():
            copy.resolve[key] = val
        return copy

    @utils.use_signature(core.AxisResolveMap)
    def resolve_axis(self, *args, **kwargs) -> Self:
        return self._set_resolve(axis=core.AxisResolveMap(*args, **kwargs))

    @utils.use_signature(core.LegendResolveMap)
    def resolve_legend(self, *args, **kwargs) -> Self:
        return self._set_resolve(legend=core.LegendResolveMap(*args, **kwargs))

    @utils.use_signature(core.ScaleResolveMap)
    def resolve_scale(self, *args, **kwargs) -> Self:
        return self._set_resolve(scale=core.ScaleResolveMap(*args, **kwargs))
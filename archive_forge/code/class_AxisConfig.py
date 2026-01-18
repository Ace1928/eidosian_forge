from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class AxisConfig(VegaLiteSchema):
    """AxisConfig schema wrapper

    Parameters
    ----------

    aria : bool, dict, :class:`ExprRef`
        A boolean flag indicating if `ARIA attributes
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ should be
        included (SVG output only). If ``false``, the "aria-hidden" attribute will be set on
        the output SVG group, removing the axis from the ARIA accessibility tree.

        **Default value:** ``true``
    bandPosition : dict, float, :class:`ExprRef`
        An interpolation fraction indicating where, for ``band`` scales, axis ticks should
        be positioned. A value of ``0`` places ticks at the left edge of their bands. A
        value of ``0.5`` places ticks in the middle of their bands.

        **Default value:** ``0.5``
    description : str, dict, :class:`ExprRef`
        A text description of this axis for `ARIA accessibility
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ (SVG output
        only). If the ``aria`` property is true, for SVG output the `"aria-label" attribute
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Techniques/Using_the_aria-label_attribute>`__
        will be set to this description. If the description is unspecified it will be
        automatically generated.
    disable : bool
        Disable axis by default.
    domain : bool
        A boolean flag indicating if the domain (the axis baseline) should be included as
        part of the axis.

        **Default value:** ``true``
    domainCap : dict, :class:`ExprRef`, :class:`StrokeCap`, Literal['butt', 'round', 'square']
        The stroke cap for the domain line's ending style. One of ``"butt"``, ``"round"`` or
        ``"square"``.

        **Default value:** ``"butt"``
    domainColor : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Color of axis domain line.

        **Default value:** ``"gray"``.
    domainDash : dict, Sequence[float], :class:`ExprRef`
        An array of alternating [stroke, space] lengths for dashed domain lines.
    domainDashOffset : dict, float, :class:`ExprRef`
        The pixel offset at which to start drawing with the domain dash array.
    domainOpacity : dict, float, :class:`ExprRef`
        Opacity of the axis domain line.
    domainWidth : dict, float, :class:`ExprRef`
        Stroke width of axis domain line

        **Default value:** ``1``
    format : str, dict, :class:`Dict`
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__.
        * If the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : str
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:**


        * ``"time"`` for temporal fields and ordinal and nominal fields with ``timeUnit``.
        * ``"number"`` for quantitative fields as well as ordinal and nominal fields without
          ``timeUnit``.
    grid : bool
        A boolean flag indicating if grid lines should be included as part of the axis

        **Default value:** ``true`` for `continuous scales
        <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__ that are not
        binned; otherwise, ``false``.
    gridCap : dict, :class:`ExprRef`, :class:`StrokeCap`, Literal['butt', 'round', 'square']
        The stroke cap for grid lines' ending style. One of ``"butt"``, ``"round"`` or
        ``"square"``.

        **Default value:** ``"butt"``
    gridColor : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, :class:`ConditionalAxisColor`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Color of gridlines.

        **Default value:** ``"lightGray"``.
    gridDash : dict, Sequence[float], :class:`ExprRef`, :class:`ConditionalAxisNumberArray`
        An array of alternating [stroke, space] lengths for dashed grid lines.
    gridDashOffset : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The pixel offset at which to start drawing with the grid dash array.
    gridOpacity : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The stroke opacity of grid (value between [0,1])

        **Default value:** ``1``
    gridWidth : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The grid width, in pixels.

        **Default value:** ``1``
    labelAlign : dict, :class:`Align`, :class:`ExprRef`, :class:`ConditionalAxisLabelAlign`, Literal['left', 'center', 'right']
        Horizontal text alignment of axis tick labels, overriding the default setting for
        the current axis orientation.
    labelAngle : dict, float, :class:`ExprRef`
        The rotation angle of the axis labels.

        **Default value:** ``-90`` for nominal and ordinal fields; ``0`` otherwise.
    labelBaseline : str, dict, :class:`ExprRef`, :class:`Baseline`, :class:`TextBaseline`, Literal['top', 'middle', 'bottom'], :class:`ConditionalAxisLabelBaseline`
        Vertical text baseline of axis tick labels, overriding the default setting for the
        current axis orientation. One of ``"alphabetic"`` (default), ``"top"``,
        ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The ``"line-top"``
        and ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but
        are calculated relative to the *lineHeight* rather than *fontSize* alone.
    labelBound : bool, dict, float, :class:`ExprRef`
        Indicates if labels should be hidden if they exceed the axis range. If ``false``
        (the default) no bounds overlap analysis is performed. If ``true``, labels will be
        hidden if they exceed the axis range by more than 1 pixel. If this property is a
        number, it specifies the pixel tolerance: the maximum amount by which a label
        bounding box may exceed the axis range.

        **Default value:** ``false``.
    labelColor : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, :class:`ConditionalAxisColor`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        The color of the tick label, can be in hex color code or regular color name.
    labelExpr : str
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    labelFlush : bool, float
        Indicates if the first and last axis labels should be aligned flush with the scale
        range. Flush alignment for a horizontal axis will left-align the first label and
        right-align the last label. For vertical axes, bottom and top text baselines are
        applied instead. If this property is a number, it also indicates the number of
        pixels by which to offset the first and last labels; for example, a value of 2 will
        flush-align the first and last labels and also push them 2 pixels outward from the
        center of the axis. The additional adjustment can sometimes help the labels better
        visually group with corresponding axis ticks.

        **Default value:** ``true`` for axis of a continuous x-scale. Otherwise, ``false``.
    labelFlushOffset : dict, float, :class:`ExprRef`
        Indicates the number of pixels by which to offset flush-adjusted labels. For
        example, a value of ``2`` will push flush-adjusted labels 2 pixels outward from the
        center of the axis. Offsets can help the labels better visually group with
        corresponding axis ticks.

        **Default value:** ``0``.
    labelFont : str, dict, :class:`ExprRef`, :class:`ConditionalAxisString`
        The font of the tick label.
    labelFontSize : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The font size of the label, in pixels.
    labelFontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`, :class:`ConditionalAxisLabelFontStyle`
        Font style of the title.
    labelFontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, :class:`ConditionalAxisLabelFontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        Font weight of axis tick labels.
    labelLimit : dict, float, :class:`ExprRef`
        Maximum allowed pixel width of axis tick labels.

        **Default value:** ``180``
    labelLineHeight : dict, float, :class:`ExprRef`
        Line height in pixels for multi-line label text or label text with ``"line-top"`` or
        ``"line-bottom"`` baseline.
    labelOffset : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        Position offset in pixels to apply to labels, in addition to tickOffset.

        **Default value:** ``0``
    labelOpacity : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The opacity of the labels.
    labelOverlap : str, bool, dict, :class:`ExprRef`, :class:`LabelOverlap`
        The strategy to use for resolving overlap of axis labels. If ``false`` (the
        default), no overlap reduction is attempted. If set to ``true`` or ``"parity"``, a
        strategy of removing every other label is used (this works well for standard linear
        axes). If set to ``"greedy"``, a linear scan of the labels is performed, removing
        any labels that overlaps with the last visible label (this often works better for
        log-scaled axes).

        **Default value:** ``true`` for non-nominal fields with non-log scales; ``"greedy"``
        for log scales; otherwise ``false``.
    labelPadding : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The padding in pixels between labels and ticks.

        **Default value:** ``2``
    labelSeparation : dict, float, :class:`ExprRef`
        The minimum separation that must be between label bounding boxes for them to be
        considered non-overlapping (default ``0`` ). This property is ignored if
        *labelOverlap* resolution is not enabled.
    labels : bool
        A boolean flag indicating if labels should be included as part of the axis.

        **Default value:** ``true``.
    maxExtent : dict, float, :class:`ExprRef`
        The maximum extent in pixels that axis ticks and labels should use. This determines
        a maximum offset value for axis titles.

        **Default value:** ``undefined``.
    minExtent : dict, float, :class:`ExprRef`
        The minimum extent in pixels that axis ticks and labels should use. This determines
        a minimum offset value for axis titles.

        **Default value:** ``30`` for y-axis; ``undefined`` for x-axis.
    offset : dict, float, :class:`ExprRef`
        The offset, in pixels, by which to displace the axis from the edge of the enclosing
        group or data rectangle.

        **Default value:** derived from the `axis config
        <https://vega.github.io/vega-lite/docs/config.html#facet-scale-config>`__ 's
        ``offset`` ( ``0`` by default)
    orient : dict, :class:`ExprRef`, :class:`AxisOrient`, Literal['top', 'bottom', 'left', 'right']
        The orientation of the axis. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``. The orientation can be used to further specialize the axis type (e.g.,
        a y-axis oriented towards the right edge of the chart).

        **Default value:** ``"bottom"`` for x-axes and ``"left"`` for y-axes.
    position : dict, float, :class:`ExprRef`
        The anchor position of the axis in pixels. For x-axes with top or bottom
        orientation, this sets the axis group x coordinate. For y-axes with left or right
        orientation, this sets the axis group y coordinate.

        **Default value** : ``0``
    style : str, Sequence[str]
        A string or array of strings indicating the name of custom styles to apply to the
        axis. A style is a named collection of axis property defined within the `style
        configuration <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. If
        style is an array, later styles will override earlier styles.

        **Default value:** (none) **Note:** Any specified style will augment the default
        style. For example, an x-axis mark with ``"style": "foo"`` will use ``config.axisX``
        and ``config.style.foo`` (the specified style ``"foo"`` has higher precedence).
    tickBand : dict, :class:`ExprRef`, Literal['center', 'extent']
        For band scales, indicates if ticks and grid lines should be placed at the
        ``"center"`` of a band (default) or at the band ``"extent"`` s to indicate intervals
    tickCap : dict, :class:`ExprRef`, :class:`StrokeCap`, Literal['butt', 'round', 'square']
        The stroke cap for the tick lines' ending style. One of ``"butt"``, ``"round"`` or
        ``"square"``.

        **Default value:** ``"butt"``
    tickColor : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, :class:`ConditionalAxisColor`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        The color of the axis's tick.

        **Default value:** ``"gray"``
    tickCount : dict, float, :class:`ExprRef`, :class:`TimeInterval`, :class:`TimeIntervalStep`, Literal['millisecond', 'second', 'minute', 'hour', 'day', 'week', 'month', 'year']
        A desired number of ticks, for axes visualizing quantitative scales. The resulting
        number may be different so that values are "nice" (multiples of 2, 5, 10) and lie
        within the underlying scale's range.

        For scales of type ``"time"`` or ``"utc"``, the tick count can instead be a time
        interval specifier. Legal string values are ``"millisecond"``, ``"second"``,
        ``"minute"``, ``"hour"``, ``"day"``, ``"week"``, ``"month"``, and ``"year"``.
        Alternatively, an object-valued interval specifier of the form ``{"interval":
        "month", "step": 3}`` includes a desired number of interval steps. Here, ticks are
        generated for each quarter (Jan, Apr, Jul, Oct) boundary.

        **Default value** : Determine using a formula ``ceil(width/40)`` for x and
        ``ceil(height/40)`` for y.
    tickDash : dict, Sequence[float], :class:`ExprRef`, :class:`ConditionalAxisNumberArray`
        An array of alternating [stroke, space] lengths for dashed tick mark lines.
    tickDashOffset : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The pixel offset at which to start drawing with the tick mark dash array.
    tickExtra : bool
        Boolean flag indicating if an extra axis tick should be added for the initial
        position of the axis. This flag is useful for styling axes for ``band`` scales such
        that ticks are placed on band boundaries rather in the middle of a band. Use in
        conjunction with ``"bandPosition": 1`` and an axis ``"padding"`` value of ``0``.
    tickMinStep : dict, float, :class:`ExprRef`
        The minimum desired step between axis ticks, in terms of scale domain values. For
        example, a value of ``1`` indicates that ticks should not be less than 1 unit apart.
        If ``tickMinStep`` is specified, the ``tickCount`` value will be adjusted, if
        necessary, to enforce the minimum step value.
    tickOffset : dict, float, :class:`ExprRef`
        Position offset in pixels to apply to ticks, labels, and gridlines.
    tickOpacity : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        Opacity of the ticks.
    tickRound : bool
        Boolean flag indicating if pixel position values should be rounded to the nearest
        integer.

        **Default value:** ``true``
    tickSize : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The size in pixels of axis ticks.

        **Default value:** ``5``
    tickWidth : dict, float, :class:`ExprRef`, :class:`ConditionalAxisNumber`
        The width, in pixels, of ticks.

        **Default value:** ``1``
    ticks : bool
        Boolean value that determines whether the axis should include ticks.

        **Default value:** ``true``
    title : str, None, :class:`Text`, Sequence[str]
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/usage/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    titleAlign : dict, :class:`Align`, :class:`ExprRef`, Literal['left', 'center', 'right']
        Horizontal text alignment of axis titles.
    titleAnchor : dict, :class:`ExprRef`, :class:`TitleAnchor`, Literal[None, 'start', 'middle', 'end']
        Text anchor position for placing axis titles.
    titleAngle : dict, float, :class:`ExprRef`
        Angle in degrees of axis titles.
    titleBaseline : str, dict, :class:`ExprRef`, :class:`Baseline`, :class:`TextBaseline`, Literal['top', 'middle', 'bottom']
        Vertical text baseline for axis titles. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the *lineHeight* rather than *fontSize*
        alone.
    titleColor : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Color of the title, can be in hex color code or regular color name.
    titleFont : str, dict, :class:`ExprRef`
        Font of the title. (e.g., ``"Helvetica Neue"`` ).
    titleFontSize : dict, float, :class:`ExprRef`
        Font size of the title.
    titleFontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`
        Font style of the title.
    titleFontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        Font weight of the title. This can be either a string (e.g ``"bold"``, ``"normal"``
        ) or a number ( ``100``, ``200``, ``300``, ..., ``900`` where ``"normal"`` = ``400``
        and ``"bold"`` = ``700`` ).
    titleLimit : dict, float, :class:`ExprRef`
        Maximum allowed pixel width of axis titles.
    titleLineHeight : dict, float, :class:`ExprRef`
        Line height in pixels for multi-line title text or title text with ``"line-top"`` or
        ``"line-bottom"`` baseline.
    titleOpacity : dict, float, :class:`ExprRef`
        Opacity of the axis title.
    titlePadding : dict, float, :class:`ExprRef`
        The padding, in pixels, between title and axis.
    titleX : dict, float, :class:`ExprRef`
        X-coordinate of the axis title relative to the axis group.
    titleY : dict, float, :class:`ExprRef`
        Y-coordinate of the axis title relative to the axis group.
    translate : dict, float, :class:`ExprRef`
        Coordinate space translation offset for axis layout. By default, axes are translated
        by a 0.5 pixel offset for both the x and y coordinates in order to align stroked
        lines with the pixel grid. However, for vector graphics output these pixel-specific
        adjustments may be undesirable, in which case translate can be changed (for example,
        to zero).

        **Default value:** ``0.5``
    values : dict, Sequence[str], Sequence[bool], Sequence[float], :class:`ExprRef`, Sequence[dict, :class:`DateTime`]
        Explicitly set the visible axis tick values.
    zindex : float
        A non-negative integer indicating the z-index of the axis. If zindex is 0, axes
        should be drawn behind all chart elements. To put them in front, set ``zindex`` to
        ``1`` or more.

        **Default value:** ``0`` (behind the marks).
    """
    _schema = {'$ref': '#/definitions/AxisConfig'}

    def __init__(self, aria: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, bandPosition: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, description: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, disable: Union[bool, UndefinedType]=Undefined, domain: Union[bool, UndefinedType]=Undefined, domainCap: Union[dict, '_Parameter', 'SchemaBase', Literal['butt', 'round', 'square'], UndefinedType]=Undefined, domainColor: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, domainDash: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, domainDashOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, domainOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, domainWidth: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, format: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, formatType: Union[str, UndefinedType]=Undefined, grid: Union[bool, UndefinedType]=Undefined, gridCap: Union[dict, '_Parameter', 'SchemaBase', Literal['butt', 'round', 'square'], UndefinedType]=Undefined, gridColor: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, gridDash: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, gridDashOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, gridOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, gridWidth: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelAlign: Union[dict, '_Parameter', 'SchemaBase', Literal['left', 'center', 'right'], UndefinedType]=Undefined, labelAngle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelBaseline: Union[str, dict, '_Parameter', 'SchemaBase', Literal['top', 'middle', 'bottom'], UndefinedType]=Undefined, labelBound: Union[bool, dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelColor: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, labelExpr: Union[str, UndefinedType]=Undefined, labelFlush: Union[bool, float, UndefinedType]=Undefined, labelFlushOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFont: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, labelLimit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelLineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelOverlap: Union[str, bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelPadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelSeparation: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labels: Union[bool, UndefinedType]=Undefined, maxExtent: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, minExtent: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, offset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, orient: Union[dict, '_Parameter', 'SchemaBase', Literal['top', 'bottom', 'left', 'right'], UndefinedType]=Undefined, position: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, style: Union[str, Sequence[str], UndefinedType]=Undefined, tickBand: Union[dict, '_Parameter', 'SchemaBase', Literal['center', 'extent'], UndefinedType]=Undefined, tickCap: Union[dict, '_Parameter', 'SchemaBase', Literal['butt', 'round', 'square'], UndefinedType]=Undefined, tickColor: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, tickCount: Union[dict, float, '_Parameter', 'SchemaBase', Literal['millisecond', 'second', 'minute', 'hour', 'day', 'week', 'month', 'year'], UndefinedType]=Undefined, tickDash: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, tickDashOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tickExtra: Union[bool, UndefinedType]=Undefined, tickMinStep: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tickOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tickOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tickRound: Union[bool, UndefinedType]=Undefined, tickSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tickWidth: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, ticks: Union[bool, UndefinedType]=Undefined, title: Union[str, None, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, titleAlign: Union[dict, '_Parameter', 'SchemaBase', Literal['left', 'center', 'right'], UndefinedType]=Undefined, titleAnchor: Union[dict, '_Parameter', 'SchemaBase', Literal[None, 'start', 'middle', 'end'], UndefinedType]=Undefined, titleAngle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleBaseline: Union[str, dict, '_Parameter', 'SchemaBase', Literal['top', 'middle', 'bottom'], UndefinedType]=Undefined, titleColor: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, titleFont: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleFontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleFontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleFontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, titleLimit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleLineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titlePadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleX: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleY: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, translate: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, values: Union[dict, '_Parameter', 'SchemaBase', Sequence[str], Sequence[bool], Sequence[float], Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, zindex: Union[float, UndefinedType]=Undefined, **kwds):
        super(AxisConfig, self).__init__(aria=aria, bandPosition=bandPosition, description=description, disable=disable, domain=domain, domainCap=domainCap, domainColor=domainColor, domainDash=domainDash, domainDashOffset=domainDashOffset, domainOpacity=domainOpacity, domainWidth=domainWidth, format=format, formatType=formatType, grid=grid, gridCap=gridCap, gridColor=gridColor, gridDash=gridDash, gridDashOffset=gridDashOffset, gridOpacity=gridOpacity, gridWidth=gridWidth, labelAlign=labelAlign, labelAngle=labelAngle, labelBaseline=labelBaseline, labelBound=labelBound, labelColor=labelColor, labelExpr=labelExpr, labelFlush=labelFlush, labelFlushOffset=labelFlushOffset, labelFont=labelFont, labelFontSize=labelFontSize, labelFontStyle=labelFontStyle, labelFontWeight=labelFontWeight, labelLimit=labelLimit, labelLineHeight=labelLineHeight, labelOffset=labelOffset, labelOpacity=labelOpacity, labelOverlap=labelOverlap, labelPadding=labelPadding, labelSeparation=labelSeparation, labels=labels, maxExtent=maxExtent, minExtent=minExtent, offset=offset, orient=orient, position=position, style=style, tickBand=tickBand, tickCap=tickCap, tickColor=tickColor, tickCount=tickCount, tickDash=tickDash, tickDashOffset=tickDashOffset, tickExtra=tickExtra, tickMinStep=tickMinStep, tickOffset=tickOffset, tickOpacity=tickOpacity, tickRound=tickRound, tickSize=tickSize, tickWidth=tickWidth, ticks=ticks, title=title, titleAlign=titleAlign, titleAnchor=titleAnchor, titleAngle=titleAngle, titleBaseline=titleBaseline, titleColor=titleColor, titleFont=titleFont, titleFontSize=titleFontSize, titleFontStyle=titleFontStyle, titleFontWeight=titleFontWeight, titleLimit=titleLimit, titleLineHeight=titleLineHeight, titleOpacity=titleOpacity, titlePadding=titlePadding, titleX=titleX, titleY=titleY, translate=translate, values=values, zindex=zindex, **kwds)
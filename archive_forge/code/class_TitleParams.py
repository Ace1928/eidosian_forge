from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TitleParams(VegaLiteSchema):
    """TitleParams schema wrapper

    Parameters
    ----------

    text : str, dict, :class:`Text`, Sequence[str], :class:`ExprRef`
        The title text.
    align : :class:`Align`, Literal['left', 'center', 'right']
        Horizontal text alignment for title text. One of ``"left"``, ``"center"``, or
        ``"right"``.
    anchor : :class:`TitleAnchor`, Literal[None, 'start', 'middle', 'end']
        The anchor position for placing the title. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with an orientation of top these anchor positions map to a
        left-, center-, or right-aligned title.

        **Default value:** ``"middle"`` for `single
        <https://vega.github.io/vega-lite/docs/spec.html>`__ and `layered
        <https://vega.github.io/vega-lite/docs/layer.html>`__ views. ``"start"`` for other
        composite views.

        **Note:** `For now <https://github.com/vega/vega-lite/issues/2875>`__, ``anchor`` is
        only customizable only for `single
        <https://vega.github.io/vega-lite/docs/spec.html>`__ and `layered
        <https://vega.github.io/vega-lite/docs/layer.html>`__ views. For other composite
        views, ``anchor`` is always ``"start"``.
    angle : dict, float, :class:`ExprRef`
        Angle in degrees of title and subtitle text.
    aria : bool, dict, :class:`ExprRef`
        A boolean flag indicating if `ARIA attributes
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ should be
        included (SVG output only). If ``false``, the "aria-hidden" attribute will be set on
        the output SVG group, removing the title from the ARIA accessibility tree.

        **Default value:** ``true``
    baseline : str, :class:`Baseline`, :class:`TextBaseline`, Literal['top', 'middle', 'bottom']
        Vertical text baseline for title and subtitle text. One of ``"alphabetic"``
        (default), ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or
        ``"line-bottom"``. The ``"line-top"`` and ``"line-bottom"`` values operate similarly
        to ``"top"`` and ``"bottom"``, but are calculated relative to the *lineHeight*
        rather than *fontSize* alone.
    color : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Text color for title text.
    dx : dict, float, :class:`ExprRef`
        Delta offset for title and subtitle text x-coordinate.
    dy : dict, float, :class:`ExprRef`
        Delta offset for title and subtitle text y-coordinate.
    font : str, dict, :class:`ExprRef`
        Font name for title text.
    fontSize : dict, float, :class:`ExprRef`
        Font size in pixels for title text.
    fontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`
        Font style for title text.
    fontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        Font weight for title text. This can be either a string (e.g ``"bold"``,
        ``"normal"`` ) or a number ( ``100``, ``200``, ``300``, ..., ``900`` where
        ``"normal"`` = ``400`` and ``"bold"`` = ``700`` ).
    frame : str, dict, :class:`ExprRef`, :class:`TitleFrame`, Literal['bounds', 'group']
        The reference frame for the anchor position, one of ``"bounds"`` (to anchor relative
        to the full bounding box) or ``"group"`` (to anchor relative to the group width or
        height).
    limit : dict, float, :class:`ExprRef`
        The maximum allowed length in pixels of title and subtitle text.
    lineHeight : dict, float, :class:`ExprRef`
        Line height in pixels for multi-line title text or title text with ``"line-top"`` or
        ``"line-bottom"`` baseline.
    offset : dict, float, :class:`ExprRef`
        The orthogonal offset in pixels by which to displace the title group from its
        position along the edge of the chart.
    orient : dict, :class:`ExprRef`, :class:`TitleOrient`, Literal['none', 'left', 'right', 'top', 'bottom']
        Default title orientation ( ``"top"``, ``"bottom"``, ``"left"``, or ``"right"`` )
    style : str, Sequence[str]
        A `mark style property <https://vega.github.io/vega-lite/docs/config.html#style>`__
        to apply to the title text mark.

        **Default value:** ``"group-title"``.
    subtitle : str, :class:`Text`, Sequence[str]
        The subtitle Text.
    subtitleColor : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Text color for subtitle text.
    subtitleFont : str, dict, :class:`ExprRef`
        Font name for subtitle text.
    subtitleFontSize : dict, float, :class:`ExprRef`
        Font size in pixels for subtitle text.
    subtitleFontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`
        Font style for subtitle text.
    subtitleFontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        Font weight for subtitle text. This can be either a string (e.g ``"bold"``,
        ``"normal"`` ) or a number ( ``100``, ``200``, ``300``, ..., ``900`` where
        ``"normal"`` = ``400`` and ``"bold"`` = ``700`` ).
    subtitleLineHeight : dict, float, :class:`ExprRef`
        Line height in pixels for multi-line subtitle text.
    subtitlePadding : dict, float, :class:`ExprRef`
        The padding in pixels between title and subtitle text.
    zindex : float
        The integer z-index indicating the layering of the title group relative to other
        axis, mark and legend groups.

        **Default value:** ``0``.
    """
    _schema = {'$ref': '#/definitions/TitleParams'}

    def __init__(self, text: Union[str, dict, '_Parameter', 'SchemaBase', Sequence[str], UndefinedType]=Undefined, align: Union['SchemaBase', Literal['left', 'center', 'right'], UndefinedType]=Undefined, anchor: Union['SchemaBase', Literal[None, 'start', 'middle', 'end'], UndefinedType]=Undefined, angle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, aria: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, baseline: Union[str, 'SchemaBase', Literal['top', 'middle', 'bottom'], UndefinedType]=Undefined, color: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, dx: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, dy: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, font: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, frame: Union[str, dict, '_Parameter', 'SchemaBase', Literal['bounds', 'group'], UndefinedType]=Undefined, limit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, lineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, offset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, orient: Union[dict, '_Parameter', 'SchemaBase', Literal['none', 'left', 'right', 'top', 'bottom'], UndefinedType]=Undefined, style: Union[str, Sequence[str], UndefinedType]=Undefined, subtitle: Union[str, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, subtitleColor: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, subtitleFont: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, subtitleFontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, subtitleFontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, subtitleFontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, subtitleLineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, subtitlePadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, zindex: Union[float, UndefinedType]=Undefined, **kwds):
        super(TitleParams, self).__init__(text=text, align=align, anchor=anchor, angle=angle, aria=aria, baseline=baseline, color=color, dx=dx, dy=dy, font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight, frame=frame, limit=limit, lineHeight=lineHeight, offset=offset, orient=orient, style=style, subtitle=subtitle, subtitleColor=subtitleColor, subtitleFont=subtitleFont, subtitleFontSize=subtitleFontSize, subtitleFontStyle=subtitleFontStyle, subtitleFontWeight=subtitleFontWeight, subtitleLineHeight=subtitleLineHeight, subtitlePadding=subtitlePadding, zindex=zindex, **kwds)
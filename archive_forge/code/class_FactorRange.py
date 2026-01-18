from __future__ import annotations
import logging # isort:skip
from collections import Counter
from math import nan
from ..core.enums import PaddingUnits, StartEnd
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import DUPLICATE_FACTORS
from ..model import Model
class FactorRange(Range):
    """ A Range of values for a categorical dimension.

    In addition to supplying ``factors`` as a keyword argument to the
    ``FactorRange`` initializer, you may also instantiate with a sequence of
    positional arguments:

    .. code-block:: python

        FactorRange("foo", "bar") # equivalent to FactorRange(factors=["foo", "bar"])

    Users will normally supply categorical values directly:

    .. code-block:: python

        p.circle(x=["foo", "bar"], ...)

    BokehJS will create a mapping from ``"foo"`` and ``"bar"`` to a numerical
    coordinate system called *synthetic coordinates*. In the simplest cases,
    factors are separated by a distance of 1.0 in synthetic coordinates,
    however the exact mapping from factors to synthetic coordinates is
    affected by he padding properties as well as whether the number of levels
    the factors have.

    Users typically do not need to worry about the details of this mapping,
    however it can be useful to fine tune positions by adding offsets. When
    supplying factors as coordinates or values, it is possible to add an
    offset in the synthetic coordinate space by adding a final number value
    to a factor tuple. For example:

    .. code-block:: python

        p.circle(x=[("foo", 0.3), ...], ...)

    will position the first circle at an ``x`` position that is offset by
    adding 0.3 to the synthetic coordinate for ``"foo"``.

    """
    factors = FactorSeq(default=[], help='\n    A sequence of factors to define this categorical range.\n\n    Factors may have 1, 2, or 3 levels. For 1-level factors, each factor is\n    simply a string. For example:\n\n    .. code-block:: python\n\n        FactorRange(factors=["sales", "marketing", "engineering"])\n\n    defines a range with three simple factors that might represent different\n    units of a business.\n\n    For 2- and 3- level factors, each factor is a tuple of strings:\n\n    .. code-block:: python\n\n        FactorRange(factors=[\n            ["2016", "sales"], ["2016", "marketing"], ["2016", "engineering"],\n            ["2017", "sales"], ["2017", "marketing"], ["2017", "engineering"],\n        ])\n\n    defines a range with six 2-level factors that might represent the three\n    business units, grouped by year.\n\n    Note that factors and sub-factors *may only be strings*.\n\n    ')
    factor_padding = Float(default=0.0, help='\n    How much padding to add in between all lowest-level factors. When\n    ``factor_padding`` is non-zero, every factor in every group will have the\n    padding value applied.\n    ')
    subgroup_padding = Float(default=0.8, help="\n    How much padding to add in between mid-level groups of factors. This\n    property only applies when the overall factors have three levels. For\n    example with:\n\n    .. code-block:: python\n\n        FactorRange(factors=[\n            ['foo', 'A', '1'],  ['foo', 'A', '2'], ['foo', 'A', '3'],\n            ['foo', 'B', '2'],\n            ['bar', 'A', '1'],  ['bar', 'A', '2']\n        ])\n\n    This property dictates how much padding to add between the three factors\n    in the `['foo', 'A']` group, and between the two factors in the the\n    [`bar`]\n    ")
    group_padding = Float(default=1.4, help='\n    How much padding to add in between top-level groups of factors. This\n    property only applies when the overall range factors have either two or\n    three levels. For example, with:\n\n    .. code-block:: python\n\n        FactorRange(factors=[["foo", "1"], ["foo", "2"], ["bar", "1"]])\n\n    The top level groups correspond to ``"foo"`` and ``"bar"``, and the\n    group padding will be applied between the factors ``["foo", "2"]`` and\n    ``["bar", "1"]``\n    ')
    range_padding = Float(default=0, help='\n    How much padding to add around the outside of computed range bounds.\n\n    When ``range_padding_units`` is set to ``"percent"``, the span of the\n    range span is expanded to make the range ``range_padding`` percent larger.\n\n    When ``range_padding_units`` is set to ``"absolute"``, the start and end\n    of the range span are extended by the amount ``range_padding``.\n    ')
    range_padding_units = Enum(PaddingUnits, default='percent', help='\n    Whether the ``range_padding`` should be interpreted as a percentage, or\n    as an absolute quantity. (default: ``"percent"``)\n    ')
    start = Readonly(Float, default=0, help='\n    The start of the range, in synthetic coordinates.\n\n    .. note::\n        Synthetic coordinates are only computed in the browser, based on the\n        factors and various padding properties. The value of ``start`` will only\n        be available in situations where bidirectional communication is\n        available (e.g. server, notebook).\n    ')
    end = Readonly(Float, default=0, help='\n    The end of the range, in synthetic coordinates.\n\n    .. note::\n        Synthetic coordinates are only computed in the browser, based on the\n        factors and various padding properties. The value of ``end`` will only\n        be available in situations where bidirectional communication is\n        available (e.g. server, notebook).\n    ')
    bounds = Nullable(MinMaxBounds(accept_datetime=False), help="\n    The bounds (in synthetic coordinates) that the range is allowed to go to.\n    Typically used to prevent the user from panning/zooming/etc away from the\n    data.\n\n    .. note::\n        Synthetic coordinates are only computed in the browser, based on the\n        factors and various padding properties. Some experimentation may be\n        required to arrive at bounds suitable for specific situations.\n\n    By default, the bounds will be None, allowing your plot to pan/zoom as far\n    as you want. If bounds are 'auto' they will be computed to be the same as\n    the start and end of the ``FactorRange``.\n    ")
    min_interval = Nullable(Float, help='\n    The level that the range is allowed to zoom in, expressed as the\n    minimum visible interval in synthetic coordinates. If set to ``None``\n    (default), the minimum interval is not bounded.\n\n    The default "width" of a category is 1.0 in synthetic coordinates.\n    However, the distance between factors is affected by the various\n    padding properties and whether or not factors are grouped.\n    ')
    max_interval = Nullable(Float, help='\n    The level that the range is allowed to zoom out, expressed as the\n    maximum visible interval in synthetic coordinates.. Note that ``bounds``\n    can impose an implicit constraint on the maximum interval as well.\n\n    The default "width" of a category is 1.0 in synthetic coordinates.\n    However, the distance between factors is affected by the various\n    padding properties and whether or not factors are grouped.\n    ')

    def __init__(self, *args, **kwargs) -> None:
        if args and 'factors' in kwargs:
            raise ValueError("'factors' keyword cannot be used with positional arguments")
        elif args:
            kwargs['factors'] = list(args)
        super().__init__(**kwargs)

    @error(DUPLICATE_FACTORS)
    def _check_duplicate_factors(self):
        dupes = [item for item, count in Counter(self.factors).items() if count > 1]
        if dupes:
            return 'duplicate factors found: %s' % ', '.join((repr(x) for x in dupes))
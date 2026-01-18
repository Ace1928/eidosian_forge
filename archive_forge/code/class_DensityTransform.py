from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class DensityTransform(Transform):
    """DensityTransform schema wrapper

    Parameters
    ----------

    density : str, :class:`FieldName`
        The data field for which to perform density estimation.
    bandwidth : float
        The bandwidth (standard deviation) of the Gaussian kernel. If unspecified or set to
        zero, the bandwidth value is automatically estimated from the input data using
        Scott’s rule.
    counts : bool
        A boolean flag indicating if the output values should be probability estimates
        (false) or smoothed counts (true).

        **Default value:** ``false``
    cumulative : bool
        A boolean flag indicating whether to produce density estimates (false) or cumulative
        density estimates (true).

        **Default value:** ``false``
    extent : Sequence[float]
        A [min, max] domain from which to sample the distribution. If unspecified, the
        extent will be determined by the observed minimum and maximum values of the density
        value field.
    groupby : Sequence[str, :class:`FieldName`]
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    maxsteps : float
        The maximum number of samples to take along the extent domain for plotting the
        density.

        **Default value:** ``200``
    minsteps : float
        The minimum number of samples to take along the extent domain for plotting the
        density.

        **Default value:** ``25``
    steps : float
        The exact number of samples to take along the extent domain for plotting the
        density. If specified, overrides both minsteps and maxsteps to set an exact number
        of uniform samples. Potentially useful in conjunction with a fixed extent to ensure
        consistent sample points for stacked densities.
    as : Sequence[str, :class:`FieldName`]
        The output fields for the sample value and corresponding density estimate.

        **Default value:** ``["value", "density"]``
    """
    _schema = {'$ref': '#/definitions/DensityTransform'}

    def __init__(self, density: Union[str, 'SchemaBase', UndefinedType]=Undefined, bandwidth: Union[float, UndefinedType]=Undefined, counts: Union[bool, UndefinedType]=Undefined, cumulative: Union[bool, UndefinedType]=Undefined, extent: Union[Sequence[float], UndefinedType]=Undefined, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, maxsteps: Union[float, UndefinedType]=Undefined, minsteps: Union[float, UndefinedType]=Undefined, steps: Union[float, UndefinedType]=Undefined, **kwds):
        super(DensityTransform, self).__init__(density=density, bandwidth=bandwidth, counts=counts, cumulative=cumulative, extent=extent, groupby=groupby, maxsteps=maxsteps, minsteps=minsteps, steps=steps, **kwds)
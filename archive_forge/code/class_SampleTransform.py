from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SampleTransform(Transform):
    """SampleTransform schema wrapper

    Parameters
    ----------

    sample : float
        The maximum number of data objects to include in the sample.

        **Default value:** ``1000``
    """
    _schema = {'$ref': '#/definitions/SampleTransform'}

    def __init__(self, sample: Union[float, UndefinedType]=Undefined, **kwds):
        super(SampleTransform, self).__init__(sample=sample, **kwds)
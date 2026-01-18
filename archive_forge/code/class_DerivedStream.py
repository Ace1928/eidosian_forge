from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class DerivedStream(Stream):
    """DerivedStream schema wrapper

    Parameters
    ----------

    stream : dict, :class:`Stream`, :class:`EventStream`, :class:`MergedStream`, :class:`DerivedStream`

    between : Sequence[dict, :class:`Stream`, :class:`EventStream`, :class:`MergedStream`, :class:`DerivedStream`]

    consume : bool

    debounce : float

    filter : str, :class:`Expr`, Sequence[str, :class:`Expr`]

    markname : str

    marktype : :class:`MarkType`, Literal['arc', 'area', 'image', 'group', 'line', 'path', 'rect', 'rule', 'shape', 'symbol', 'text', 'trail']

    throttle : float

    """
    _schema = {'$ref': '#/definitions/DerivedStream'}

    def __init__(self, stream: Union[dict, 'SchemaBase', UndefinedType]=Undefined, between: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, consume: Union[bool, UndefinedType]=Undefined, debounce: Union[float, UndefinedType]=Undefined, filter: Union[str, 'SchemaBase', Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, markname: Union[str, UndefinedType]=Undefined, marktype: Union['SchemaBase', Literal['arc', 'area', 'image', 'group', 'line', 'path', 'rect', 'rule', 'shape', 'symbol', 'text', 'trail'], UndefinedType]=Undefined, throttle: Union[float, UndefinedType]=Undefined, **kwds):
        super(DerivedStream, self).__init__(stream=stream, between=between, consume=consume, debounce=debounce, filter=filter, markname=markname, marktype=marktype, throttle=throttle, **kwds)
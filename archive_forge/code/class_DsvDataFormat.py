from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class DsvDataFormat(DataFormat):
    """DsvDataFormat schema wrapper

    Parameters
    ----------

    delimiter : str
        The delimiter between records. The delimiter must be a single character (i.e., a
        single 16-bit code unit); so, ASCII delimiters are fine, but emoji delimiters are
        not.
    parse : dict, None, :class:`Parse`
        If set to ``null``, disable type inference based on the spec and only use type
        inference based on the data. Alternatively, a parsing directive object can be
        provided for explicit data types. Each property of the object corresponds to a field
        name, and the value to the desired data type (one of ``"number"``, ``"boolean"``,
        ``"date"``, or null (do not parse the field)). For example, ``"parse":
        {"modified_on": "date"}`` parses the ``modified_on`` field in each input record a
        Date value.

        For ``"date"``, we parse data based using JavaScript's `Date.parse()
        <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/parse>`__.
        For Specific date formats can be provided (e.g., ``{foo: "date:'%m%d%Y'"}`` ), using
        the `d3-time-format syntax <https://github.com/d3/d3-time-format#locale_format>`__.
        UTC date format parsing is supported similarly (e.g., ``{foo: "utc:'%m%d%Y'"}`` ).
        See more about `UTC time
        <https://vega.github.io/vega-lite/docs/timeunit.html#utc>`__
    type : str
        Type of input data: ``"json"``, ``"csv"``, ``"tsv"``, ``"dsv"``.

        **Default value:**  The default format type is determined by the extension of the
        file URL. If no extension is detected, ``"json"`` will be used by default.
    """
    _schema = {'$ref': '#/definitions/DsvDataFormat'}

    def __init__(self, delimiter: Union[str, UndefinedType]=Undefined, parse: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, type: Union[str, UndefinedType]=Undefined, **kwds):
        super(DsvDataFormat, self).__init__(delimiter=delimiter, parse=parse, type=type, **kwds)
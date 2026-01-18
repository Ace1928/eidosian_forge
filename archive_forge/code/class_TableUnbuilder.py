import collections
import enum
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables.otConverters import (
from fontTools.misc.roundTools import otRound
class TableUnbuilder:

    def __init__(self, callbackTable=None):
        if callbackTable is None:
            callbackTable = {}
        self._callbackTable = callbackTable

    def unbuild(self, table):
        assert isinstance(table, BaseTable)
        source = {}
        callbackKey = (type(table),)
        if isinstance(table, FormatSwitchingBaseTable):
            source['Format'] = int(table.Format)
            callbackKey += (table.Format,)
        for converter in table.getConverters():
            if isinstance(converter, ComputedInt):
                continue
            value = getattr(table, converter.name)
            enumClass = getattr(converter, 'enumClass', None)
            if enumClass:
                source[converter.name] = value.name.lower()
            elif isinstance(converter, Struct):
                if converter.repeat:
                    source[converter.name] = [self.unbuild(v) for v in value]
                else:
                    source[converter.name] = self.unbuild(value)
            elif isinstance(converter, SimpleValue):
                source[converter.name] = value
            else:
                raise NotImplementedError("Don't know how unbuild {value!r} with {converter!r}")
        source = self._callbackTable.get(callbackKey, lambda s: s)(source)
        return source
import collections
import enum
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables.otConverters import (
from fontTools.misc.roundTools import otRound
class TableBuilder:
    """
    Helps to populate things derived from BaseTable from maps, tuples, etc.

    A table of lifecycle callbacks may be provided to add logic beyond what is possible
    based on otData info for the target class. See BuildCallbacks.
    """

    def __init__(self, callbackTable=None):
        if callbackTable is None:
            callbackTable = {}
        self._callbackTable = callbackTable

    def _convert(self, dest, field, converter, value):
        enumClass = getattr(converter, 'enumClass', None)
        if enumClass:
            if isinstance(value, enumClass):
                pass
            elif isinstance(value, str):
                try:
                    value = getattr(enumClass, value.upper())
                except AttributeError:
                    raise ValueError(f'{value} is not a valid {enumClass}')
            else:
                value = enumClass(value)
        elif isinstance(converter, IntValue):
            value = otRound(value)
        elif isinstance(converter, FloatValue):
            value = float(value)
        elif isinstance(converter, Struct):
            if converter.repeat:
                if _isNonStrSequence(value):
                    value = [self.build(converter.tableClass, v) for v in value]
                else:
                    value = [self.build(converter.tableClass, value)]
                setattr(dest, converter.repeat, len(value))
            else:
                value = self.build(converter.tableClass, value)
        elif callable(converter):
            value = converter(value)
        setattr(dest, field, value)

    def build(self, cls, source):
        assert issubclass(cls, BaseTable)
        if isinstance(source, cls):
            return source
        callbackKey = (cls,)
        fmt = None
        if issubclass(cls, FormatSwitchingBaseTable):
            fmt, source = _split_format(cls, source)
            callbackKey = (cls, fmt)
        dest = self._callbackTable.get((BuildCallback.CREATE_DEFAULT,) + callbackKey, lambda: cls())()
        assert isinstance(dest, cls)
        convByName = _assignable(cls.convertersByName)
        skippedFields = set()
        if issubclass(cls, FormatSwitchingBaseTable):
            dest.Format = fmt
            convByName = _assignable(convByName[dest.Format])
            skippedFields.add('Format')
        if _isNonStrSequence(source):
            assert len(source) <= len(convByName), f'Sequence of {len(source)} too long for {cls}; expected <= {len(convByName)} values'
            source = dict(zip(convByName.keys(), source))
        dest, source = self._callbackTable.get((BuildCallback.BEFORE_BUILD,) + callbackKey, lambda d, s: (d, s))(dest, source)
        if isinstance(source, collections.abc.Mapping):
            for field, value in source.items():
                if field in skippedFields:
                    continue
                converter = convByName.get(field, None)
                if not converter:
                    raise ValueError(f'Unrecognized field {field} for {cls}; expected one of {sorted(convByName.keys())}')
                self._convert(dest, field, converter, value)
        else:
            dest = self.build(cls, (source,))
        for field, conv in convByName.items():
            if not hasattr(dest, field) and isinstance(conv, OptionalValue):
                setattr(dest, field, conv.DEFAULT)
        dest = self._callbackTable.get((BuildCallback.AFTER_BUILD,) + callbackKey, lambda d: d)(dest)
        return dest
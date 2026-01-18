import typing as t
class TimestampNTZType(DataType):

    @classmethod
    def typeName(cls) -> str:
        return 'timestamp_ntz'
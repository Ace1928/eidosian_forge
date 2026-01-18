import math
import typing
import uuid
class StringConvertor(Convertor[str]):
    regex = '[^/]+'

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        value = str(value)
        assert '/' not in value, 'May not contain path separators'
        assert value, 'Must not be empty'
        return value
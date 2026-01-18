import math
import typing
import uuid
class PathConvertor(Convertor[str]):
    regex = '.*'

    def convert(self, value: str) -> str:
        return str(value)

    def to_string(self, value: str) -> str:
        return str(value)
import typing as t
class DataTypeWithLength(DataType):

    def __init__(self, length: int):
        self.length = length

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.length})'

    def __str__(self) -> str:
        return f'{self.typeName()}({self.length})'
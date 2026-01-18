import typing as t
class DecimalType(DataType):

    def __init__(self, precision: int=10, scale: int=0):
        self.precision = precision
        self.scale = scale

    def simpleString(self) -> str:
        return f'decimal({self.precision}, {self.scale})'

    def jsonValue(self) -> str:
        return f'decimal({self.precision}, {self.scale})'

    def __repr__(self) -> str:
        return f'DecimalType({self.precision}, {self.scale})'
import math
import typing
import uuid
class FloatConvertor(Convertor[float]):
    regex = '[0-9]+(\\.[0-9]+)?'

    def convert(self, value: str) -> float:
        return float(value)

    def to_string(self, value: float) -> str:
        value = float(value)
        assert value >= 0.0, 'Negative floats are not supported'
        assert not math.isnan(value), 'NaN values are not supported'
        assert not math.isinf(value), 'Infinite values are not supported'
        return ('%0.20f' % value).rstrip('0').rstrip('.')
from param import Parameter, _is_number
class Aspect(Parameter):
    """
    A Parameter type to validate aspect ratios. Supports numeric values
    and auto.
    """

    def __init__(self, default=None, allow_None=True, **params):
        super().__init__(default=default, allow_None=allow_None, **params)
        self._validate(default)

    def _validate(self, val):
        self._validate_value(val, self.allow_None)

    def _validate_value(self, val, allow_None):
        if val is None and allow_None or val == 'auto' or _is_number(val):
            return
        raise ValueError(f"Aspect parameter {self.name!r} only takes numeric values or the literal 'auto'.")
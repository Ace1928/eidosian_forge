import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
class Validated(Property):

    def __init__(self, *args: Func, validators: Optional[List[Validator]]=None, **kwargs: Func) -> None:
        super().__init__(*args, **kwargs)
        if validators is None:
            validators = []
        self.validators = validators

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(value, type(self)):
            for validator in self.validators:
                validator(self, value)
        super().__set__(instance, value)
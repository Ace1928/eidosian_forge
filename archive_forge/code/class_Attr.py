import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
class Attr(Typed, JSONLinked):

    def getter(self, fget: Func) -> Func:
        prop = type(self)(fget, self.fset, json_path=self.path_or_name, validators=self.validators)
        prop.name = self.name
        return prop

    def setter(self, fset: Func) -> Func:
        prop = type(self)(self.fget, fset, json_path=self.path_or_name, validators=self.validators)
        prop.name = self.name
        return prop
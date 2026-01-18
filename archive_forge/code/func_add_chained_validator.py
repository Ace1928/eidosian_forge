import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
@declarative.classinstancemethod
def add_chained_validator(self, cls, validator):
    if self is not None:
        if self.chained_validators is cls.chained_validators:
            self.chained_validators = cls.chained_validators[:]
        self.chained_validators.append(validator)
    else:
        cls.chained_validators.append(validator)
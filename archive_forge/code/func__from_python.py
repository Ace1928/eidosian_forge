from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
@classmethod
def _from_python(cls, data: dict):
    """ Generate an instance from the plain python representation """
    self = cls('')
    for key in ['full_implications', 'beta_triggers', 'prereq']:
        d = defaultdict(set)
        d.update(data[key])
        setattr(self, key, d)
    self.beta_rules = data['beta_rules']
    self.defined_facts = set(data['defined_facts'])
    return self
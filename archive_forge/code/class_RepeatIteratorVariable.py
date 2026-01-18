from typing import List, Optional
from ..exc import unimplemented
from .base import VariableTracker
from .constant import ConstantVariable
class RepeatIteratorVariable(IteratorVariable):

    def __init__(self, item: VariableTracker, **kwargs):
        super().__init__(**kwargs)
        self.item = item

    def next_variables(self, tx):
        return (self.item.clone(), self)
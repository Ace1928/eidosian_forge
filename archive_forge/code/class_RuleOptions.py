from typing import Optional, Tuple, ClassVar, Sequence
from .utils import Serialize
class RuleOptions(Serialize):
    __serialize_fields__ = ('keep_all_tokens', 'expand1', 'priority', 'template_source', 'empty_indices')
    keep_all_tokens: bool
    expand1: bool
    priority: Optional[int]
    template_source: Optional[str]
    empty_indices: Tuple[bool, ...]

    def __init__(self, keep_all_tokens: bool=False, expand1: bool=False, priority: Optional[int]=None, template_source: Optional[str]=None, empty_indices: Tuple[bool, ...]=()) -> None:
        self.keep_all_tokens = keep_all_tokens
        self.expand1 = expand1
        self.priority = priority
        self.template_source = template_source
        self.empty_indices = empty_indices

    def __repr__(self):
        return 'RuleOptions(%r, %r, %r, %r)' % (self.keep_all_tokens, self.expand1, self.priority, self.template_source)
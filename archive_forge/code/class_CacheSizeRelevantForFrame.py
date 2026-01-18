import logging
import types
import weakref
from dataclasses import dataclass
from . import config
@dataclass
class CacheSizeRelevantForFrame:
    """
    We track the number of cache entries that have same id_match objects as the
    given frame.

    TODO(janimesh) - Consider adding a map from tuple_of_match_ids to count -
    https://github.com/pytorch/pytorch/pull/107496#discussion_r1304564682 - this
    could be useful for debugging as well.
    """
    num_cache_entries: int = 0
    num_cache_entries_in_bucket: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        return self.will_compilation_exceed_bucket(limit) or self.will_compilation_exceed_total()

    def will_compilation_exceed_total(self) -> bool:
        return self.num_cache_entries >= config.accumulated_cache_size_limit

    def will_compilation_exceed_bucket(self, limit: int) -> bool:
        return self.num_cache_entries_in_bucket >= limit
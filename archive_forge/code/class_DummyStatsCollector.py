import logging
import pprint
from typing import TYPE_CHECKING, Any, Dict, Optional
from scrapy import Spider
class DummyStatsCollector(StatsCollector):

    def get_value(self, key: str, default: Any=None, spider: Optional[Spider]=None) -> Any:
        return default

    def set_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        pass

    def set_stats(self, stats: StatsT, spider: Optional[Spider]=None) -> None:
        pass

    def inc_value(self, key: str, count: int=1, start: int=0, spider: Optional[Spider]=None) -> None:
        pass

    def max_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        pass

    def min_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        pass
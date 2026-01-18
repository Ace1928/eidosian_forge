from __future__ import annotations
import copy
import typing as t
from . import common
def _clean_found_objs(self) -> None:
    if self.foaf_name:
        title = self.foaf_name[-1]
    else:
        title = ''
    for url in self.agent_feeds:
        obj = common.SuperDict({'url': url, 'title': title})
        self.group_objs.append(('feeds', obj))
    for url in self.agent_lists:
        obj = common.SuperDict({'url': url, 'title': title})
        self.group_objs.append(('lists', obj))
    for url in self.agent_opps:
        obj = common.SuperDict({'url': url, 'title': title})
        self.group_objs.append(('opportunities', obj))
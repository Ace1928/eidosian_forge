from __future__ import annotations
from collections import namedtuple
def _populate():
    for k, v in TAGS_V2.items():
        TAGS[k] = v[0]
        if len(v) == 4:
            for sk, sv in v[3].items():
                TAGS[k, sv] = sk
        TAGS_V2[k] = TagInfo(k, *v)
    for tags in TAGS_V2_GROUPS.values():
        for k, v in tags.items():
            tags[k] = TagInfo(k, *v)
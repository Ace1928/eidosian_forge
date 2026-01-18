import copy
from typing import Any
from six.moves.urllib.parse import urljoin, urlparse
from extruct.dublincore import get_lower_attrib
def _uopengraph(extracted, with_og_array=False):
    out = []
    for obj in extracted:
        properties = list(obj['properties'])
        flattened: dict[Any, Any] = {}
        for k, v in properties:
            if k not in flattened.keys():
                flattened[k] = v
            elif v and v.strip():
                if not with_og_array:
                    if not flattened[k] or not flattened[k].strip():
                        flattened[k] = v
                elif isinstance(flattened[k], list):
                    flattened[k].append(v)
                elif flattened[k] and flattened[k].strip():
                    flattened[k] = [flattened[k], v]
                else:
                    flattened[k] = v
        t = flattened.pop('og:type', None)
        if t:
            flattened['@type'] = t
        flattened['@context'] = obj['namespace']
        out.append(flattened)
    return out
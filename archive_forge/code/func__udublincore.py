import copy
from typing import Any
from six.moves.urllib.parse import urljoin, urlparse
from extruct.dublincore import get_lower_attrib
def _udublincore(extracted):
    out = []
    extracted_cpy = copy.deepcopy(extracted)
    for obj in extracted_cpy:
        context = obj.pop('namespaces', None)
        obj['@context'] = context
        elements = obj['elements']
        for element in elements:
            for key, value in element.items():
                if get_lower_attrib(value) == 'type':
                    obj['@type'] = element['content']
                    obj['elements'].remove(element)
                    break
        out.append(obj)
    return out
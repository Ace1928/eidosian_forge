import typing as ty
import urllib.parse
from openstack import exceptions
from openstack import resource
@classmethod
def _get_next_link(cls, uri, response, data, marker, limit, total_yielded):
    next_link = None
    params: ty.Dict[str, ty.Union[ty.List[str], str]] = {}
    if isinstance(data, dict):
        links = data.get('links')
        if links:
            next_link = links.get('next')
        total = data.get('metadata', {}).get('total_count')
        if total:
            total_count = int(total)
            if total_count <= total_yielded:
                return (None, params)
    if next_link:
        parts = urllib.parse.urlparse(next_link)
        query_params = urllib.parse.parse_qs(parts.query)
        params.update(query_params)
        next_link = urllib.parse.urljoin(next_link, parts.path)
    if not next_link and limit:
        next_link = uri
        params['marker'] = marker
        params['limit'] = limit
    return (next_link, params)
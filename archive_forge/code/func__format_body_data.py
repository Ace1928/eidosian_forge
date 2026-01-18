import copy
from urllib import parse as urlparse
from magnumclient.common.apiclient import base
def _format_body_data(self, body, response_key):
    if response_key:
        try:
            data = body[response_key]
        except KeyError:
            return []
    else:
        data = body
    if not isinstance(data, list):
        data = [data]
    return data
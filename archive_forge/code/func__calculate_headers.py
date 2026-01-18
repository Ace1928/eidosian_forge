import typing as ty
from openstack import exceptions
from openstack import resource
def _calculate_headers(self, metadata):
    headers = {}
    for key in metadata:
        if key in self._system_metadata.keys():
            header = self._system_metadata[key]
        elif key in self._system_metadata.values():
            header = key
        elif key.startswith(self._custom_metadata_prefix):
            header = key
        else:
            header = self._custom_metadata_prefix + key
        headers[header] = metadata[key]
    return headers
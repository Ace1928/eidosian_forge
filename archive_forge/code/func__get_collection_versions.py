from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
def _get_collection_versions(self, requirement):
    """Helper for get_collection_versions.

        Yield api, version pairs for all APIs,
        and reraise the last error if no valid API was found.
        """
    if self._offline:
        return []
    found_api = False
    last_error = None
    api_lookup_order = (requirement.src,) if isinstance(requirement.src, GalaxyAPI) else self._apis
    for api in api_lookup_order:
        try:
            versions = api.get_collection_versions(requirement.namespace, requirement.name)
        except GalaxyError as api_err:
            last_error = api_err
        except Exception as unknown_err:
            display.warning('Skipping Galaxy server {server!s}. Got an unexpected error when getting available versions of collection {fqcn!s}: {err!s}'.format(server=api.api_server, fqcn=requirement.fqcn, err=to_text(unknown_err)))
            last_error = unknown_err
        else:
            found_api = True
            for version in versions:
                yield (api, version)
    if not found_api and last_error is not None:
        raise last_error
import logging
from optparse import Values
from typing import Any, Dict, List
from pip._vendor.packaging.markers import default_environment
from pip._vendor.rich import print_json
from pip import __version__
from pip._internal.cli import cmdoptions
from pip._internal.cli.req_command import Command
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.utils.compat import stdlib_pkgs
from pip._internal.utils.urls import path_to_url
def _dist_to_dict(self, dist: BaseDistribution) -> Dict[str, Any]:
    res: Dict[str, Any] = {'metadata': dist.metadata_dict, 'metadata_location': dist.info_location}
    direct_url = dist.direct_url
    if direct_url is not None:
        res['direct_url'] = direct_url.to_dict()
    else:
        editable_project_location = dist.editable_project_location
        if editable_project_location is not None:
            res['direct_url'] = {'url': path_to_url(editable_project_location), 'dir_info': {'editable': True}}
    installer = dist.installer
    if dist.installer:
        res['installer'] = installer
    if dist.installed_with_dist_info:
        res['requested'] = dist.requested
    return res
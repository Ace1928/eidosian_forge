import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
def _get_server_version_range(client):
    version = client.versions.get_current()
    if not hasattr(version, 'max_version') or not version.max_version:
        return (APIVersion(), APIVersion())
    return (APIVersion(version.min_version), APIVersion(version.max_version))
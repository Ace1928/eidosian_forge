from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
def _BuildAssetTypeFilterFromKind(kind_list):
    """Get assetType Filter from KRM Kind list."""
    if not kind_list:
        return None
    name_translator = resource_name_translator.ResourceNameTranslator()
    kind_filters = []
    for kind in kind_list:
        krm_kind = kind
        if '/' in kind:
            _, krm_kind = kind.split('/')
        matching_kind_objects = name_translator.find_krmkinds_by_kind(krm_kind)
        try:
            for kind_obj in matching_kind_objects:
                meta_resource = name_translator.get_resource(krm_kind=kind_obj)
                kind_filters.append(meta_resource.asset_inventory_type)
        except resource_name_translator.ResourceIdentifierNotFoundError:
            continue
    return kind_filters
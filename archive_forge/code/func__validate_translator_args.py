from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
def _validate_translator_args(asset_inventory_type=None, krm_kind=None, collection_name=None):
    """Validates that arguments passed to the translator are correctly passed."""
    args_specified = sum((bool(identifier) for identifier in [asset_inventory_type, krm_kind, collection_name]))
    if args_specified > 1:
        raise ResourceNameTranslatorError('Must specify only one [asset_inventory_type, krm_kind, collection_name]')
    if args_specified < 1:
        raise ResourceNameTranslatorError('Must specify at least one of [asset_inventory_type, krm_kind, collection_name]')
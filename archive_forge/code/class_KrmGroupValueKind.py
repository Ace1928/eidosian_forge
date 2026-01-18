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
class KrmGroupValueKind(object):
    """Value class for KRM Group Value Kind Data."""

    def __init__(self, kind, group, bulk_export_supported, export_supported, iam_supported, version=None, resource_name_format=None):
        self.kind = kind
        self.group = group
        self.version = version
        self.bulk_export_supported = bulk_export_supported
        self.export_supported = export_supported
        self.iam_supported = iam_supported
        self.resource_name_format = resource_name_format

    def AsDict(self):
        """Convert to Config Connector compatible dict format."""
        gvk = collections.OrderedDict()
        output = collections.OrderedDict()
        gvk['Group'] = self.group
        gvk['Kind'] = self.kind
        gvk['Version'] = self.version or ''
        output['GVK'] = gvk
        output['ResourceNameFormat'] = self.resource_name_format or ''
        output['SupportsBulkExport'] = self.bulk_export_supported
        output['SupportsExport'] = self.export_supported
        output['SupportsIAM'] = self.iam_supported
        return output

    def __str__(self):
        return yaml.dump(self.AsDict(), round_trip=True)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        if not isinstance(o, KrmGroupValueKind):
            return False
        return self.kind == o.kind and self.group == o.group and (self.version == o.version) and (self.bulk_export_supported == o.bulk_export_supported) and (self.export_supported == o.export_supported) and (self.iam_supported == o.iam_supported) and (self.resource_name_format == o.resource_name_format)

    def __hash__(self):
        return sum(map(hash, [self.kind, self.group, self.version, self.bulk_export_supported, self.export_supported, self.iam_supported, self.resource_name_format]))
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def _GenerateAuxiliaryVersionsVersionList(aux_versions):
    return _GenerateAdditionalProperties({'aux-' + version.replace('.', '-'): {'version': version} for version in aux_versions})
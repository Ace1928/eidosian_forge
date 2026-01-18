from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastore import rewrite_backend
class OperationsRewriteBackend(rewrite_backend.OperationsRewriteBackend):
    _KEY_MAPPING = {'^collectionIds$': 'metadata.collectionIds'}
    _KEY_OPERAND_MAPPING = {}
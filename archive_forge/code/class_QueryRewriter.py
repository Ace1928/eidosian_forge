from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_expr_rewrite
class QueryRewriter(resource_expr_rewrite.Backend):
    """Resource filter expression rewriter."""

    def RewriteGlobal(self, call):
        """Rewrites global restriction in call.

    Args:
      call: A list of resource_lex._TransformCall objects. In this case the list
        has one element that is a global restriction with a global_restriction
        property that is the restriction substring to match.

    Returns:
      The global restriction rewrite which is simply the global_restriction
      string.
    """
        return call.global_restriction

    def RewriteTerm(self, key, op, operand, key_type):
        """Rewrites <key op operand>."""
        del key_type
        if op in ('~',):
            raise QueryOperatorNotSupported('The [{}] operator is not supported in cloud resource search queries.'.format(op))
        values = operand if isinstance(operand, list) else [operand]
        if key == 'project':
            key = 'selfLink'
            values = ['/projects/{}/'.format(value) for value in values]
        elif key == '@type':
            collections = values
            values = []
            for collection in collections:
                if collection.startswith(CLOUD_RESOURCE_SEARCH_COLLECTION + '.'):
                    values.append(collection[len(CLOUD_RESOURCE_SEARCH_COLLECTION) + 1:])
                else:
                    try:
                        values.append(RESOURCE_TYPES[collection])
                    except KeyError:
                        raise CollectionNotIndexed('Collection [{}] not indexed for search.'.format(collection))
        parts = ['{key}{op}{value}'.format(key=key, op=op, value=self.Quote(value)) for value in values]
        expr = ' OR '.join(parts)
        if len(parts) > 1:
            expr = '( ' + expr + ' )'
        return expr
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
import six
def EntitiesType(entities):
    """Validates entities input and turns it into an entities dict.

  Valid entities inputs are:
    str of comma separated entities
    list of entities
    map from entities to synonyms

  Args:
    entities: entities input
  Returns:
    dict mapping from entities to synonyms
  Raises:
    InvalidArgumentException: If the entities input is invalid.
  """
    if isinstance(entities, six.text_type):
        entities = arg_parsers.ArgList()(entities)
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, six.text_type):
                break
        else:
            return [{'value': entity, 'synonyms': [entity]} for entity in entities]
    if isinstance(entities, dict):
        for entity, synonyms in entities.items():
            if not isinstance(synonyms, list):
                break
        else:
            return [{'value': entity, 'synonyms': synonyms} for entity, synonyms in entities.items()]
    raise exceptions.InvalidArgumentException('Entities must be a list of entities or a map from entities to a list ofsynonyms.')
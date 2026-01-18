from ..language.location import get_location
from .base import GraphQLError
class GraphQLSyntaxError(GraphQLError):

    def __init__(self, source, position, description):
        location = get_location(source, position)
        super(GraphQLSyntaxError, self).__init__(message=u'Syntax Error {} ({}:{}) {}\n\n{}'.format(source.name, location.line, location.column, description, highlight_source_at_location(source, location)), source=source, positions=[position])
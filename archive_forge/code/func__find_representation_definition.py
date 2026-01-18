import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def _find_representation_definition(self, media_type=None):
    """Get the most appropriate representation definition.

        If media_type is provided, the most appropriate definition is
        the definition of the representation of that media type.

        If this resource is bound to a representation, the most
        appropriate definition is the definition of that
        representation. Otherwise, the most appropriate definition is
        the definition of the representation served in response to a
        standard GET.

        :param media_type: Media type of the definition to find. Must
            be present unless the resource is bound to a
            representation.

        :raise NoBoundRepresentationError: If this resource is not
            bound to a representation and media_type was not provided.

        :return: A RepresentationDefinition
        """
    if self.representation is not None:
        definition = self.representation_definition.resolve_definition()
    elif media_type is not None:
        definition = self.get_representation_definition(media_type)
    else:
        raise NoBoundRepresentationError('Resource is not bound to any representation, and no media media type was specified.')
    return definition.resolve_definition()
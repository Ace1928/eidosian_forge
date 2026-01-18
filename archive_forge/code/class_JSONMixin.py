import json
from pydeck.types.base import PydeckType
class JSONMixin(object):

    def __repr__(self):
        """
        Override of string representation method to return a JSON-ified version of the
        Deck object.
        """
        return serialize(self)

    def to_json(self):
        """
        Return a JSON-ified version of the Deck object.
        """
        return serialize(self)
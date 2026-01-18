import json
from troveclient import base
from troveclient import common
class ConfigurationParameter(base.Resource):
    """Configuration Parameter."""

    def __repr__(self):
        return '<ConfigurationParameter: %s>' % self.__dict__
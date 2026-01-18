import re
from collections import namedtuple
class DocumentedShape(_DocumentedShape):
    """Use this class to inject new shapes into a model for documentation"""

    def __new__(cls, name, type_name, documentation, metadata=None, members=None, required_members=None):
        if metadata is None:
            metadata = []
        if members is None:
            members = []
        if required_members is None:
            required_members = []
        return super().__new__(cls, name, type_name, documentation, metadata, members, required_members)
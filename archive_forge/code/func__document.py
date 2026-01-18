import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document(self, section, value, comments, path, shape):
    """
        :param section: The section to add the docs to.

        :param value: The input / output values representing the parameters that
                      are included in the example.

        :param comments: The dictionary containing all the comments to be
                         applied to the example.

        :param path: A list describing where the documenter is in traversing the
                     parameters. This is used to find the equivalent location
                     in the comments dictionary.
        """
    if isinstance(value, dict):
        self._document_dict(section, value, comments, path, shape)
    elif isinstance(value, list):
        self._document_list(section, value, comments, path, shape)
    elif isinstance(value, numbers.Number):
        self._document_number(section, value, path)
    elif shape and shape.type_name == 'timestamp':
        self._document_datetime(section, value, path)
    else:
        self._document_str(section, value, path)
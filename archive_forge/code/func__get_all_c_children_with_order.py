import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def _get_all_c_children_with_order(self):
    if len(self.c_child_order) > 0:
        yield from self.c_child_order
    else:
        for _, values in iter(self.__class__.c_children.items()):
            yield values[0]
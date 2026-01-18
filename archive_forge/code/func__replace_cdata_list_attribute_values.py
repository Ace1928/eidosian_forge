from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
def _replace_cdata_list_attribute_values(self, tag_name, attrs):
    """When an attribute value is associated with a tag that can
        have multiple values for that attribute, convert the string
        value to a list of strings.

        Basically, replaces class="foo bar" with class=["foo", "bar"]

        NOTE: This method modifies its input in place.

        :param tag_name: The name of a tag.
        :param attrs: A dictionary containing the tag's attributes.
           Any appropriate attribute values will be modified in place.
        """
    if not attrs:
        return attrs
    if self.cdata_list_attributes:
        universal = self.cdata_list_attributes.get('*', [])
        tag_specific = self.cdata_list_attributes.get(tag_name.lower(), None)
        for attr in list(attrs.keys()):
            if attr in universal or (tag_specific and attr in tag_specific):
                value = attrs[attr]
                if isinstance(value, str):
                    values = nonwhitespace_re.findall(value)
                else:
                    values = value
                attrs[attr] = values
    return attrs
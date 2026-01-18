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
class RequestDefinition(WADLBase, HasParametersMixin):
    """A wrapper around the description of the request invoking a method."""

    def __init__(self, method, request_tag):
        """Initialize with a <request> tag.

        :param resource: The resource to which this request can be sent.
        :param request_tag: An ElementTree <request> tag.
        """
        self.method = method
        self.resource = self.method.resource
        self.application = self.resource.application
        self.tag = request_tag

    @property
    def query_params(self):
        """Return the query parameters for this method."""
        return self.params(['query'])

    @property
    def representations(self):
        for definition in self.tag.findall(wadl_xpath('representation')):
            yield RepresentationDefinition(self.application, self.resource, definition)

    def get_representation_definition(self, media_type=None):
        """Return the appropriate representation definition."""
        for representation in self.representations:
            if media_type is None or representation.media_type == media_type:
                return representation
        return None

    def representation(self, media_type=None, param_values=None, **kw_param_values):
        """Build a representation to be sent along with this request.

        :return: A 2-tuple of (media_type, representation).
        """
        definition = self.get_representation_definition(media_type)
        if definition is None:
            raise TypeError('Cannot build representation of media type %s' % media_type)
        return definition.bind(param_values, **kw_param_values)

    def build_url(self, param_values=None, **kw_param_values):
        """Return the request URL to use to invoke this method."""
        validated_values = self.validate_param_values(self.query_params, param_values, **kw_param_values)
        url = self.resource.url
        if len(validated_values) > 0:
            if '?' in url:
                append = '&'
            else:
                append = '?'
            url += append + urlencode(sorted(validated_values.items()))
        return url
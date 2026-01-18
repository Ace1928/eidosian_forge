from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebsiteValue(_messages.Message):
    """The bucket's website configuration, controlling how the service
    behaves when accessing bucket contents as a web site. See the Static
    Website Examples for more information.

    Fields:
      mainPageSuffix: If the requested object path is missing, the service
        will ensure the path has a trailing '/', append this suffix, and
        attempt to retrieve the resulting object. This allows the creation of
        index.html objects to represent directory pages.
      notFoundPage: If the requested object path is missing, and any
        mainPageSuffix object is missing, if applicable, the service will
        return the named object from this bucket as the content for a 404 Not
        Found result.
    """
    mainPageSuffix = _messages.StringField(1)
    notFoundPage = _messages.StringField(2)
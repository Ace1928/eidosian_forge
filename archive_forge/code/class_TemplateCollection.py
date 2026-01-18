import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
class TemplateCollection:
    """Represent a collection of :class:`.Template` objects,
    identifiable via URI.

    A :class:`.TemplateCollection` is linked to the usage of
    all template tags that address other templates, such
    as ``<%include>``, ``<%namespace>``, and ``<%inherit>``.
    The ``file`` attribute of each of those tags refers
    to a string URI that is passed to that :class:`.Template`
    object's :class:`.TemplateCollection` for resolution.

    :class:`.TemplateCollection` is an abstract class,
    with the usual default implementation being :class:`.TemplateLookup`.

    """

    def has_template(self, uri):
        """Return ``True`` if this :class:`.TemplateLookup` is
        capable of returning a :class:`.Template` object for the
        given ``uri``.

        :param uri: String URI of the template to be resolved.

        """
        try:
            self.get_template(uri)
            return True
        except exceptions.TemplateLookupException:
            return False

    def get_template(self, uri, relativeto=None):
        """Return a :class:`.Template` object corresponding to the given
        ``uri``.

        The default implementation raises
        :class:`.NotImplementedError`. Implementations should
        raise :class:`.TemplateLookupException` if the given ``uri``
        cannot be resolved.

        :param uri: String URI of the template to be resolved.
        :param relativeto: if present, the given ``uri`` is assumed to
         be relative to this URI.

        """
        raise NotImplementedError()

    def filename_to_uri(self, uri, filename):
        """Convert the given ``filename`` to a URI relative to
        this :class:`.TemplateCollection`."""
        return uri

    def adjust_uri(self, uri, filename):
        """Adjust the given ``uri`` based on the calling ``filename``.

        When this method is called from the runtime, the
        ``filename`` parameter is taken directly to the ``filename``
        attribute of the calling template. Therefore a custom
        :class:`.TemplateCollection` subclass can place any string
        identifier desired in the ``filename`` parameter of the
        :class:`.Template` objects it constructs and have them come back
        here.

        """
        return uri
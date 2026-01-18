import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
class AggregatingLocator(Locator):
    """
    This class allows you to chain and/or merge a list of locators.
    """

    def __init__(self, *locators, **kwargs):
        """
        Initialise an instance.

        :param locators: The list of locators to search.
        :param kwargs: Passed to the superclass constructor,
                       except for:
                       * merge - if False (the default), the first successful
                         search from any of the locators is returned. If True,
                         the results from all locators are merged (this can be
                         slow).
        """
        self.merge = kwargs.pop('merge', False)
        self.locators = locators
        super(AggregatingLocator, self).__init__(**kwargs)

    def clear_cache(self):
        super(AggregatingLocator, self).clear_cache()
        for locator in self.locators:
            locator.clear_cache()

    def _set_scheme(self, value):
        self._scheme = value
        for locator in self.locators:
            locator.scheme = value
    scheme = property(Locator.scheme.fget, _set_scheme)

    def _get_project(self, name):
        result = {}
        for locator in self.locators:
            d = locator.get_project(name)
            if d:
                if self.merge:
                    files = result.get('urls', {})
                    digests = result.get('digests', {})
                    result.update(d)
                    df = result.get('urls')
                    if files and df:
                        for k, v in files.items():
                            if k in df:
                                df[k] |= v
                            else:
                                df[k] = v
                    dd = result.get('digests')
                    if digests and dd:
                        dd.update(digests)
                else:
                    if self.matcher is None:
                        found = True
                    else:
                        found = False
                        for k in d:
                            if self.matcher.match(k):
                                found = True
                                break
                    if found:
                        result = d
                        break
        return result

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        result = set()
        for locator in self.locators:
            try:
                result |= locator.get_distribution_names()
            except NotImplementedError:
                pass
        return result
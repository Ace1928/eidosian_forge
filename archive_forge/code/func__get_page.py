import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
def _get_page(self, *args, **kwargs):
    """
        Return an instance of a single page.

        This hook can be used by subclasses to use an alternative to the
        standard :cls:`Page` object.
        """
    return Page(*args, **kwargs)
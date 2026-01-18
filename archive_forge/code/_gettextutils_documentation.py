import copy
import gettext
import locale
import os
from oslo_i18n import _factory
from oslo_i18n import _locale
A version of gettext.find using a cache.

    gettext.find looks for mo files on the disk using os.path.exists. Those
    don't tend to change over time, but the system calls pile up with a
    long-running service. This caches the result so that we return the same mo
    files, and only call find once per domain.
    
from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _patch_multiple(target, spec=None, create=False, spec_set=None, autospec=None, new_callable=None, **kwargs):
    """Perform multiple patches in a single call. It takes the object to be
    patched (either as an object or a string to fetch the object by importing)
    and keyword arguments for the patches::

        with patch.multiple(settings, FIRST_PATCH='one', SECOND_PATCH='two'):
            ...

    Use `DEFAULT` as the value if you want `patch.multiple` to create
    mocks for you. In this case the created mocks are passed into a decorated
    function by keyword, and a dictionary is returned when `patch.multiple` is
    used as a context manager.

    `patch.multiple` can be used as a decorator, class decorator or a context
    manager. The arguments `spec`, `spec_set`, `create`,
    `autospec` and `new_callable` have the same meaning as for `patch`. These
    arguments will be applied to *all* patches done by `patch.multiple`.

    When used as a class decorator `patch.multiple` honours `patch.TEST_PREFIX`
    for choosing which methods to wrap.
    """
    if type(target) in (unicode, str):
        getter = lambda: _importer(target)
    else:
        getter = lambda: target
    if not kwargs:
        raise ValueError('Must supply at least one keyword argument with patch.multiple')
    items = list(kwargs.items())
    attribute, new = items[0]
    patcher = _patch(getter, attribute, new, spec, create, spec_set, autospec, new_callable, {})
    patcher.attribute_name = attribute
    for attribute, new in items[1:]:
        this_patcher = _patch(getter, attribute, new, spec, create, spec_set, autospec, new_callable, {})
        this_patcher.attribute_name = attribute
        patcher.additional_patchers.append(this_patcher)
    return patcher
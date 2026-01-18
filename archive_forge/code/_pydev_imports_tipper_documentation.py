import inspect
import os.path
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked
from inspect import getfullargspec

        @param obj_to_complete: the object from where we should get the completions
        @param dir_comps: if passed, we should not 'dir' the object and should just iterate those passed as kwonly_arg parameter
        @param getattr: the way to get kwonly_arg given object from the obj_to_complete (used for the completer)
        @param filter: kwonly_arg callable that receives the name and decides if it should be appended or not to the results
        @return: list of tuples, so that each tuple represents kwonly_arg completion with:
            name, doc, args, type (from the TYPE_* constants)
    
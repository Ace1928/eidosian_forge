from __future__ import print_function, absolute_import
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
from shibokensupport.signature.mapping import update_mapping, namespace
from textwrap import dedent

errorhandler.py

This module handles the TypeError messages which were previously
produced by the generated C code.

This version is at least consistent with the signatures, which
are created by the same module.

Experimentally, we are trying to guess those errors which are
just the wrong number of elements in an iterator.
At the moment, it is unclear whether the information given is
enough to produce a useful ValueError.

This matter will be improved in a later version.

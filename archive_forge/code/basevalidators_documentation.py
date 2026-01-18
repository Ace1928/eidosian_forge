import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module

        Helper to add indexes to a validator's name, call validate_coerce on
        a value, then restore the original validator name.

        This makes sure that if a validation error message is raised, the
        property name the user sees includes the index(es) of the offending
        element.

        Parameters
        ----------
        val:
            A value to be validated
        validator
            A validator
        inds
            List of one or more non-negative integers that represent the
            nested index of the value being validated
        Returns
        -------
        val
            validated value

        Raises
        ------
        ValueError
            if val fails validation
        
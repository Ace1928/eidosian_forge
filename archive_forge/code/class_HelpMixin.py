from __future__ import annotations
import ast
import contextlib
import inspect
import re
import types
from typing import TYPE_CHECKING, Any, Final, cast
import streamlit
from streamlit.proto.DocString_pb2 import DocString as DocStringProto
from streamlit.proto.DocString_pb2 import Member as MemberProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_runner import (
from streamlit.runtime.secrets import Secrets
from streamlit.string_util import is_mem_address_str
class HelpMixin:

    @gather_metrics('help')
    def help(self, obj: Any=streamlit) -> DeltaGenerator:
        """Display help and other information for a given object.

        Depending on the type of object that is passed in, this displays the
        object's name, type, value, signature, docstring, and member variables,
        methods — as well as the values/docstring of members and methods.

        Parameters
        ----------
        obj : any
            The object whose information should be displayed. If left
            unspecified, this call will display help for Streamlit itself.

        Example
        -------

        Don't remember how to initialize a dataframe? Try this:

        >>> import streamlit as st
        >>> import pandas
        >>>
        >>> st.help(pandas.DataFrame)

        .. output::
            https://doc-string.streamlit.app/
            height: 700px

        Want to quickly check what data type is output by a certain function?
        Try:

        >>> import streamlit as st
        >>>
        >>> x = my_poorly_documented_function()
        >>> st.help(x)

        Want to quickly inspect an object? No sweat:

        >>> class Dog:
        >>>   '''A typical dog.'''
        >>>
        >>>   def __init__(self, breed, color):
        >>>     self.breed = breed
        >>>     self.color = color
        >>>
        >>>   def bark(self):
        >>>     return 'Woof!'
        >>>
        >>>
        >>> fido = Dog('poodle', 'white')
        >>>
        >>> st.help(fido)

        .. output::
            https://doc-string1.streamlit.app/
            height: 300px

        And if you're using Magic, you can get help for functions, classes,
        and modules without even typing ``st.help``:

        >>> import streamlit as st
        >>> import pandas
        >>>
        >>> # Get help for Pandas read_csv:
        >>> pandas.read_csv
        >>>
        >>> # Get help for Streamlit itself:
        >>> st

        .. output::
            https://doc-string2.streamlit.app/
            height: 700px
        """
        doc_string_proto = DocStringProto()
        _marshall(doc_string_proto, obj)
        return self.dg._enqueue('doc_string', doc_string_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)
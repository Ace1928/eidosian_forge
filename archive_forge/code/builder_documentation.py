import doctest
import unittest
from genshi.builder import Element, tag
from genshi.core import Attrs, Markup, Stream
from genshi.input import XML

        Verify that if an attribute value is given as an int (or some other
        non-string type), it is coverted to a string when the stream is
        generated.
        
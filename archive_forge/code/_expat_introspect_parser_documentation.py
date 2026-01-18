from xml.parsers.expat import ParserCreate
from dbus.exceptions import IntrospectionParserException
Return a dict mapping ``interface.method`` strings to the
    concatenation of all their 'in' parameters, and mapping
    ``interface.signal`` strings to the concatenation of all their
    parameters.

    Example output::

        {
            'com.example.SignalEmitter.OneString': 's',
            'com.example.MethodImplementor.OneInt32Argument': 'i',
        }

    :Parameters:
        `data` : str
            The introspection XML. Must be an 8-bit string of UTF-8.
    
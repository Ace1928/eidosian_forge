import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def MessageToRepr(msg, multiline=False, **kwargs):
    """Return a repr-style string for a protorpc message.

    protorpc.Message.__repr__ does not return anything that could be considered
    python code. Adding this function lets us print a protorpc message in such
    a way that it could be pasted into code later, and used to compare against
    other things.

    Args:
      msg: protorpc.Message, the message to be repr'd.
      multiline: bool, True if the returned string should have each field
          assignment on its own line.
      **kwargs: {str:str}, Additional flags for how to format the string.

    Known **kwargs:
      shortstrings: bool, True if all string values should be
          truncated at 100 characters, since when mocking the contents
          typically don't matter except for IDs, and IDs are usually
          less than 100 characters.
      no_modules: bool, True if the long module name should not be printed with
          each type.

    Returns:
      str, A string of valid python (assuming the right imports have been made)
      that recreates the message passed into this function.

    """
    indent = kwargs.get('indent', 0)

    def IndentKwargs(kwargs):
        kwargs = dict(kwargs)
        kwargs['indent'] = kwargs.get('indent', 0) + 4
        return kwargs
    if isinstance(msg, list):
        s = '['
        for item in msg:
            if multiline:
                s += '\n' + ' ' * (indent + 4)
            s += MessageToRepr(item, multiline=multiline, **IndentKwargs(kwargs)) + ','
        if multiline:
            s += '\n' + ' ' * indent
        s += ']'
        return s
    if isinstance(msg, messages.Message):
        s = type(msg).__name__ + '('
        if not kwargs.get('no_modules'):
            s = msg.__module__ + '.' + s
        names = sorted([field.name for field in msg.all_fields()])
        for name in names:
            field = msg.field_by_name(name)
            if multiline:
                s += '\n' + ' ' * (indent + 4)
            value = getattr(msg, field.name)
            s += field.name + '=' + MessageToRepr(value, multiline=multiline, **IndentKwargs(kwargs)) + ','
        if multiline:
            s += '\n' + ' ' * indent
        s += ')'
        return s
    if isinstance(msg, six.string_types):
        if kwargs.get('shortstrings') and len(msg) > 100:
            msg = msg[:100]
    if isinstance(msg, datetime.datetime):

        class SpecialTZInfo(datetime.tzinfo):

            def __init__(self, offset):
                super(SpecialTZInfo, self).__init__()
                self.offset = offset

            def __repr__(self):
                s = 'TimeZoneOffset(' + repr(self.offset) + ')'
                if not kwargs.get('no_modules'):
                    s = 'apitools.base.protorpclite.util.' + s
                return s
        msg = datetime.datetime(msg.year, msg.month, msg.day, msg.hour, msg.minute, msg.second, msg.microsecond, SpecialTZInfo(msg.tzinfo.utcoffset(0)))
    return repr(msg)
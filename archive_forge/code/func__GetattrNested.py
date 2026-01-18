from apitools.base.py import encoding
import six
def _GetattrNested(message, attribute):
    """Gets a possibly nested attribute.

    Same as getattr() if attribute is a string;
    if attribute is a tuple, returns the nested attribute referred to by
    the fields in the tuple as if they were a dotted accessor path.

    (ex _GetattrNested(msg, ('foo', 'bar', 'baz')) gets msg.foo.bar.baz
    """
    if isinstance(attribute, six.string_types):
        return getattr(message, attribute)
    elif len(attribute) == 0:
        return message
    else:
        return _GetattrNested(getattr(message, attribute[0]), attribute[1:])
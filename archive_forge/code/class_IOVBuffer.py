import enum
import typing
class IOVBuffer(typing.NamedTuple):
    """A buffer to pass as a list to :meth:`wrap_iov()`.

    Defines the buffer inside a list that is passed to :meth:`wrap_iov()`. A list of these buffers are also returned in
    the `IOVUnwrapResult` under the `buffers` attribute.

    On SSPI only a buffer of the type `header`, `trailer`, or `padding` can be auto allocated. On GSSAPI all buffers
    can be auto allocated when `data=True` but the behaviour behind this is dependent on the mech it is run for.

    On the output from the `*_iov` functions the data is the bytes buffer or `None` if the buffer wasn't set. When used
    as an input to the `*_iov` functions the data can be the buffer bytes, the length of buffer to allocate or a bool
    to state whether the buffer is auto allocated or not.
    """
    type: BufferType
    data: typing.Optional[typing.Union[bytes, int, bool]]
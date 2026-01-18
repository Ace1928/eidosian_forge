from __future__ import annotations
from streamlit import util
class StreamlitAPIException(MarkdownFormattedException):
    """Base class for Streamlit API exceptions.

    An API exception should be thrown when user code interacts with the
    Streamlit API incorrectly. (That is, when we throw an exception as a
    result of a user's malformed `st.foo` call, it should be a
    StreamlitAPIException or subclass.)

    When displaying these exceptions on the frontend, we strip Streamlit
    entries from the stack trace so that the user doesn't see a bunch of
    noise related to Streamlit internals.

    """

    def __repr__(self) -> str:
        return util.repr_(self)
from __future__ import annotations
from streamlit import util
class StreamlitModuleNotFoundError(StreamlitAPIWarning):
    """Print a pretty message when a Streamlit command requires a dependency
    that is not one of our core dependencies."""

    def __init__(self, module_name, *args):
        message = f'This Streamlit command requires module "{module_name}" to be installed.'
        super().__init__(message, *args)
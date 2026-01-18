from __future__ import annotations
from streamlit import util
class StreamlitDeprecationWarning(StreamlitAPIWarning):
    """Used to display a warning.

    Note that this should not be "raised", but passed to st.exception
    instead.
    """

    def __init__(self, config_option, msg, *args):
        message = "\n{0}\n\nYou can disable this warning by disabling the config option:\n`{1}`\n\n```\nst.set_option('{1}', False)\n```\nor in your `.streamlit/config.toml`\n```\n[deprecation]\n{2} = false\n```\n    ".format(msg, config_option, config_option.split('.')[1])
        super().__init__(message, *args)

    def __repr__(self) -> str:
        return util.repr_(self)
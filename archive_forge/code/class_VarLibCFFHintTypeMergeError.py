import textwrap
class VarLibCFFHintTypeMergeError(VarLibCFFMergeError):
    """Raised when a CFF glyph cannot be merged because of hint type differences."""

    def __init__(self, hint_type, cmd_index, m_index, default_type, glyph_name):
        error_msg = f"Glyph '{glyph_name}': '{hint_type}' at index {cmd_index} in master index {m_index} differs from the default font hint type '{default_type}'"
        self.args = (error_msg,)
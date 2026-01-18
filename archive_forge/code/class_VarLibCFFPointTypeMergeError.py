import textwrap
class VarLibCFFPointTypeMergeError(VarLibCFFMergeError):
    """Raised when a CFF glyph cannot be merged because of point type differences."""

    def __init__(self, point_type, pt_index, m_index, default_type, glyph_name):
        error_msg = f"Glyph '{glyph_name}': '{point_type}' at point index {pt_index} in master index {m_index} differs from the default font point type '{default_type}'"
        self.args = (error_msg,)
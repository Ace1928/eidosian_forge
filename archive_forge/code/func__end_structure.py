from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_default
def _end_structure(self, section, start, end):
    if not section.available_sections:
        section.clear_text()
        section.write(start + end)
        self._end_nested_param(section)
    else:
        end_bracket_section = section.add_new_section('ending-bracket')
        self._end_nested_param(end_bracket_section, end)
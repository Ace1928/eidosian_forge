import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document_dict(self, section, value, comments, path, shape, top_level=False):
    dict_section = section.add_new_section('dict-value')
    self._start_nested_value(dict_section, '{')
    for key, val in value.items():
        path.append('.%s' % key)
        item_section = dict_section.add_new_section(key)
        item_section.style.new_line()
        item_comment = self._get_comment(path, comments)
        if item_comment:
            item_section.write(item_comment)
            item_section.style.new_line()
        item_section.write("'%s': " % key)
        item_shape = None
        if shape:
            if shape.type_name == 'structure':
                item_shape = shape.members.get(key)
            elif shape.type_name == 'map':
                item_shape = shape.value
        self._document(item_section, val, comments, path, item_shape)
        path.pop()
    dict_section_end = dict_section.add_new_section('ending-brace')
    self._end_nested_value(dict_section_end, '}')
    if not top_level:
        dict_section_end.write(',')
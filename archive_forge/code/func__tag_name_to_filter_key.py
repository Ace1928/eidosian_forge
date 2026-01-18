from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.six import string_types
def _tag_name_to_filter_key(tag_name):
    return f'tag:{tag_name}'
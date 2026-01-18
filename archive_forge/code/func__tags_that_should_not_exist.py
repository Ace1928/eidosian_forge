from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def _tags_that_should_not_exist(self, resource, tags):
    existing_tags = self.get_tags(resource)
    return [tag for tag in existing_tags if tag not in tags]
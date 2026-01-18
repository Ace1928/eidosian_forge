from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import string_types
from ansible.playbook.attribute import FieldAttribute
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.template import is_template
from ansible.utils.display import Display
from jinja2.nativetypes import NativeEnvironment
class CollectionSearch:
    collections = FieldAttribute(isa='list', listof=string_types, priority=100, default=_ensure_default_collection, always_post_validate=True, static=True)

    def _load_collections(self, attr, ds):
        ds = self.get_validated_value('collections', self.fattributes.get('collections'), ds, None)
        _ensure_default_collection(collection_list=ds)
        if not ds:
            return None
        env = NativeEnvironment()
        for collection_name in ds:
            if is_template(collection_name, env):
                display.warning('"collections" is not templatable, but we found: %s, it will not be templated and will be used "as is".' % collection_name)
        return ds
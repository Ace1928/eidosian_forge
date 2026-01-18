from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class StructureShape(Shape):

    @CachedProperty
    def members(self):
        members = self._shape_model.get('members', self.MAP_TYPE())
        shape_members = self.MAP_TYPE()
        for name, shape_ref in members.items():
            shape_members[name] = self._resolve_shape_ref(shape_ref)
        return shape_members

    @CachedProperty
    def event_stream_name(self):
        for member_name, member in self.members.items():
            if member.serialization.get('eventstream'):
                return member_name
        return None

    @CachedProperty
    def error_code(self):
        if not self.metadata.get('exception', False):
            return None
        error_metadata = self.metadata.get('error', {})
        code = error_metadata.get('code')
        if code:
            return code
        return self.name

    @CachedProperty
    def is_document_type(self):
        return self.metadata.get('document', False)

    @CachedProperty
    def is_tagged_union(self):
        return self.metadata.get('union', False)
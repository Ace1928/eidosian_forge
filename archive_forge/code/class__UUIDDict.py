import uuid
class _UUIDDict(dict):

    def _uuidize(self):
        if '_uuid' not in self or self['_uuid'] is None:
            self['_uuid'] = uuid.uuid4()

    @property
    def uuid(self):
        self._uuidize()
        return self['_uuid']

    @uuid.setter
    def uuid(self, value):
        self['_uuid'] = value
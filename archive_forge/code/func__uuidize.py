import uuid
def _uuidize(self):
    if '_uuid' not in self or self['_uuid'] is None:
        self['_uuid'] = uuid.uuid4()
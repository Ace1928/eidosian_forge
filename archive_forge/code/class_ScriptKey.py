import uuid
class ScriptKey:

    def __init__(self, id=None):
        self._id = id or uuid.uuid4()

    @property
    def id(self):
        return self._id

    def __eq__(self, other):
        return self._id == other

    def __repr__(self) -> str:
        return f'ScriptKey(id={self.id})'
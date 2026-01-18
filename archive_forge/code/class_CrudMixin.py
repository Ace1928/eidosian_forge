import uuid
from designateclient.tests import base
class CrudMixin:
    path_prefix = None

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        return kwargs

    def stub_entity(self, method, parts=None, entity=None, id=None, **kwargs):
        if entity:
            kwargs['json'] = entity
        if not parts:
            parts = [self.RESOURCE]
            if self.path_prefix:
                parts.insert(0, self.path_prefix)
        if id:
            if not parts:
                parts = []
            parts.append(id)
        self.stub_url(method, parts=parts, **kwargs)

    def assertList(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for i in expected:
            self.assertIn(i, actual)
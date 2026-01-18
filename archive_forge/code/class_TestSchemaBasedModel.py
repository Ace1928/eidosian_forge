import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
class TestSchemaBasedModel(testtools.TestCase):

    def setUp(self):
        super(TestSchemaBasedModel, self).setUp()
        self.model = warlock.model_factory(_SCHEMA.raw(), base_class=schemas.SchemaBasedModel)

    def test_patch_should_replace_missing_core_properties(self):
        obj = {'name': 'fred'}
        original = self.model(obj)
        original['color'] = 'red'
        patch = original.patch
        expected = '[{"path": "/color", "value": "red", "op": "replace"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_add_extra_properties(self):
        obj = {'name': 'fred'}
        original = self.model(obj)
        original['weight'] = '10'
        patch = original.patch
        expected = '[{"path": "/weight", "value": "10", "op": "add"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_replace_extra_properties(self):
        obj = {'name': 'fred', 'weight': '10'}
        original = self.model(obj)
        original['weight'] = '22'
        patch = original.patch
        expected = '[{"path": "/weight", "value": "22", "op": "replace"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_remove_extra_properties(self):
        obj = {'name': 'fred', 'weight': '10'}
        original = self.model(obj)
        del original['weight']
        patch = original.patch
        expected = '[{"path": "/weight", "op": "remove"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_remove_core_properties(self):
        obj = {'name': 'fred', 'color': 'red'}
        original = self.model(obj)
        del original['color']
        patch = original.patch
        expected = '[{"path": "/color", "op": "remove"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_add_missing_custom_properties(self):
        obj = {'name': 'fred'}
        original = self.model(obj)
        original['shape'] = 'circle'
        patch = original.patch
        expected = '[{"path": "/shape", "value": "circle", "op": "add"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_replace_custom_properties(self):
        obj = {'name': 'fred', 'shape': 'circle'}
        original = self.model(obj)
        original['shape'] = 'square'
        patch = original.patch
        expected = '[{"path": "/shape", "value": "square", "op": "replace"}]'
        self.assertTrue(compare_json_patches(patch, expected))

    def test_patch_should_replace_tags(self):
        obj = {'name': 'fred'}
        original = self.model(obj)
        original['tags'] = ['tag1', 'tag2']
        patch = original.patch
        expected = '[{"path": "/tags", "value": ["tag1", "tag2"], "op": "replace"}]'
        self.assertTrue(compare_json_patches(patch, expected))
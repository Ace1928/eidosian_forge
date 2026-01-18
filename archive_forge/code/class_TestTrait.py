import uuid
from osc_placement.tests.functional import base
class TestTrait(base.BaseTestCase):
    VERSION = '1.6'

    def test_list_traits(self):
        self.assertTrue(len(self.trait_list()) > 0)

    def test_list_associated_traits(self):
        self.trait_create(TRAIT)
        rp = self.resource_provider_create()
        self.resource_provider_trait_set(rp['uuid'], TRAIT)
        self.assertIn(TRAIT, {t['name'] for t in self.trait_list(associated=True)})

    def test_list_traits_startswith(self):
        self.trait_create(TRAIT)
        rp = self.resource_provider_create()
        self.resource_provider_trait_set(rp['uuid'], TRAIT)
        traits = {t['name'] for t in self.trait_list(name='startswith:' + TRAIT)}
        self.assertEqual(1, len(traits))
        self.assertIn(TRAIT, traits)

    def test_list_traits_startswith_unknown_trait(self):
        traits = {t['name'] for t in self.trait_list(name='startswith:CUSTOM_FOO')}
        self.assertEqual(0, len(traits))

    def test_list_traits_in(self):
        self.trait_create(TRAIT)
        rp = self.resource_provider_create()
        self.resource_provider_trait_set(rp['uuid'], TRAIT)
        traits = {t['name'] for t in self.trait_list(name='in:' + TRAIT)}
        self.assertEqual(1, len(traits))
        self.assertIn(TRAIT, traits)

    def test_list_traits_in_unknown_trait(self):
        traits = {t['name'] for t in self.trait_list(name='in:CUSTOM_FOO')}
        self.assertEqual(0, len(traits))

    def test_show_trait(self):
        self.trait_create(TRAIT)
        self.assertEqual({'name': TRAIT}, self.trait_show(TRAIT))

    def test_fail_show_unknown_trait(self):
        self.assertCommandFailed('HTTP 404', self.trait_show, 'UNKNOWN')

    def test_set_multiple_traits(self):
        self.trait_create(TRAIT + '1')
        self.trait_create(TRAIT + '2')
        rp = self.resource_provider_create()
        self.resource_provider_trait_set(rp['uuid'], TRAIT + '1', TRAIT + '2')
        traits = {t['name'] for t in self.resource_provider_trait_list(rp['uuid'])}
        self.assertEqual(2, len(traits))

    def test_set_known_and_unknown_traits(self):
        self.trait_create(TRAIT)
        rp = self.resource_provider_create()
        self.assertCommandFailed('No such trait', self.resource_provider_trait_set, rp['uuid'], TRAIT, TRAIT + '1')
        self.assertEqual([], self.resource_provider_trait_list(rp['uuid']))

    def test_delete_traits_from_provider(self):
        self.trait_create(TRAIT)
        rp = self.resource_provider_create()
        self.resource_provider_trait_set(rp['uuid'], TRAIT)
        traits = {t['name'] for t in self.resource_provider_trait_list(rp['uuid'])}
        self.assertEqual(1, len(traits))
        self.assertIn(TRAIT, traits)
        self.resource_provider_trait_delete(rp['uuid'])
        traits = {t['name'] for t in self.resource_provider_trait_list(rp['uuid'])}
        self.assertEqual(0, len(traits))

    def test_delete_trait(self):
        self.trait_create(TRAIT)
        self.trait_delete(TRAIT)
        self.assertCommandFailed('HTTP 404', self.trait_show, TRAIT)

    def test_fail_rp_trait_list_unknown_uuid(self):
        self.assertCommandFailed('No resource provider', self.resource_provider_trait_list, 123)
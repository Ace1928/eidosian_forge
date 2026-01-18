from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class TestImageMembershipFactory(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageMembershipFactory, self).setUp()
        self.factory = FakeImageMembershipFactory()

    def test_proxy_plain(self):
        proxy_factory = proxy.ImageMembershipFactory(self.factory)
        self.factory.result = 'tyrion'
        membership = proxy_factory.new_image_member('jaime', 'cersei')
        self.assertEqual('tyrion', membership)
        self.assertEqual('jaime', self.factory.image)
        self.assertEqual('cersei', self.factory.member_id)

    def test_proxy_wrapped_membership(self):
        proxy_factory = proxy.ImageMembershipFactory(self.factory, proxy_class=FakeProxy, proxy_kwargs={'a': 1})
        self.factory.result = 'tyrion'
        membership = proxy_factory.new_image_member('jaime', 'cersei')
        self.assertIsInstance(membership, FakeProxy)
        self.assertEqual('tyrion', membership.base)
        self.assertEqual({'a': 1}, membership.kwargs)
        self.assertEqual('jaime', self.factory.image)
        self.assertEqual('cersei', self.factory.member_id)

    def test_proxy_wrapped_image(self):
        proxy_factory = proxy.ImageMembershipFactory(self.factory, proxy_class=FakeProxy)
        self.factory.result = 'tyrion'
        image = FakeProxy('jaime')
        membership = proxy_factory.new_image_member(image, 'cersei')
        self.assertIsInstance(membership, FakeProxy)
        self.assertIsInstance(self.factory.image, FakeProxy)
        self.assertEqual('cersei', self.factory.member_id)

    def test_proxy_both_wrapped(self):

        class FakeProxy2(FakeProxy):
            pass
        proxy_factory = proxy.ImageMembershipFactory(self.factory, proxy_class=FakeProxy, proxy_kwargs={'b': 2})
        self.factory.result = 'tyrion'
        image = FakeProxy2('jaime')
        membership = proxy_factory.new_image_member(image, 'cersei')
        self.assertIsInstance(membership, FakeProxy)
        self.assertEqual('tyrion', membership.base)
        self.assertEqual({'b': 2}, membership.kwargs)
        self.assertIsInstance(self.factory.image, FakeProxy2)
        self.assertEqual('cersei', self.factory.member_id)
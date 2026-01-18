from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _main_test_helper(self, argv, status='active', image_mock=True):
    with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
        image = FakeImage(status=status)
        mock_get.return_value = image
        with mock.patch.object(cached_images.CacheController, '_enforce') as e:
            with mock.patch('glance.image_cache.ImageCache') as ic:
                cc = cached_images.CacheController()
                cc.cache = ic
                c_calls = []
                c_calls += argv[0].split(',')
                for call in c_calls:
                    mock.patch.object(ic, call)
                test_call = getattr(cc, argv[1])
                new_policy = argv[2]
                args = []
                if len(argv) == 4:
                    args = argv[3:]
                test_call(self.req, *args)
                if image_mock:
                    e.assert_called_once_with(self.req, image=image, new_policy=new_policy)
                else:
                    e.assert_called_once_with(self.req, new_policy=new_policy)
                mcs = []
                for method in ic.method_calls:
                    mcs.append(str(method))
                for call in c_calls:
                    if args == []:
                        args.append('')
                    elif args[0] and (not args[0].endswith("'")):
                        args[0] = "'" + args[0] + "'"
                    self.assertIn('call.' + call + '(' + args[0] + ')', mcs)
                self.assertEqual(len(c_calls), len(mcs))
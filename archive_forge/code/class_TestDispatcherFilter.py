from oslo_utils import timeutils
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher as notify_dispatcher
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestDispatcherFilter(test_utils.BaseTestCase):
    scenarios = [('publisher_id_match', dict(filter_rule=dict(publisher_id='^compute.*'), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=True)), ('publisher_id_nomatch', dict(filter_rule=dict(publisher_id='^compute.*'), publisher_id='network01.manager', event_type='instance.create.start', context={}, match=False)), ('event_type_match', dict(filter_rule=dict(event_type='^instance\\.create'), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=True)), ('event_type_nomatch', dict(filter_rule=dict(event_type='^instance\\.delete'), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=False)), ('event_type_not_string', dict(filter_rule=dict(event_type='^instance\\.delete'), publisher_id='compute01.manager', event_type=['instance.swim', 'instance.fly'], context={}, match=False)), ('context_match', dict(filter_rule=dict(context={'user': '^adm'}), publisher_id='compute01.manager', event_type='instance.create.start', context={'user': 'admin'}, match=True)), ('context_key_missing', dict(filter_rule=dict(context={'user': '^adm'}), publisher_id='compute01.manager', event_type='instance.create.start', context={'project': 'admin'}, metadata={}, match=False)), ('metadata_match', dict(filter_rule=dict(metadata={'message_id': '^99'}), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=True)), ('metadata_key_missing', dict(filter_rule=dict(metadata={'user': '^adm'}), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=False)), ('payload_match', dict(filter_rule=dict(payload={'state': '^active$'}), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=True)), ('payload_no_match', dict(filter_rule=dict(payload={'state': '^deleted$'}), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=False)), ('payload_key_missing', dict(filter_rule=dict(payload={'user': '^adm'}), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=False)), ('payload_value_none', dict(filter_rule=dict(payload={'virtual_size': '2048'}), publisher_id='compute01.manager', event_type='instance.create.start', context={}, match=False)), ('mix_match', dict(filter_rule=dict(event_type='^instance\\.create', publisher_id='^compute', context={'user': '^adm'}), publisher_id='compute01.manager', event_type='instance.create.start', context={'user': 'admin'}, match=True))]

    def test_filters(self):
        notification_filter = oslo_messaging.NotificationFilter(**self.filter_rule)
        endpoint = mock.Mock(spec=['info'], filter_rule=notification_filter)
        dispatcher = notify_dispatcher.NotificationDispatcher([endpoint], serializer=None)
        message = {'payload': {'state': 'active', 'virtual_size': None}, 'priority': 'info', 'publisher_id': self.publisher_id, 'event_type': self.event_type, 'timestamp': '2014-03-03 18:21:04.369234', 'message_id': '99863dda-97f0-443a-a0c1-6ed317b7fd45'}
        incoming = mock.Mock(ctxt=self.context, message=message)
        dispatcher.dispatch(incoming)
        if self.match:
            self.assertEqual(1, endpoint.info.call_count)
        else:
            self.assertEqual(0, endpoint.info.call_count)
import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def _set_notification(self, hit_type, transport, destination, request_type, event_types=None, test_event_type=None):
    """
        Common operation to set notification or send a test event
        notification for a specified HIT type
        """
    params = {'HITTypeId': hit_type}
    notification_params = {'Destination': destination, 'Transport': transport, 'Version': boto.mturk.notification.NotificationMessage.NOTIFICATION_VERSION, 'Active': True}
    if event_types:
        self.build_list_params(notification_params, event_types, 'EventType')
    notification_rest_params = {}
    num = 1
    for key in notification_params:
        notification_rest_params['Notification.%d.%s' % (num, key)] = notification_params[key]
    params.update(notification_rest_params)
    if test_event_type:
        params.update({'TestEventType': test_event_type})
    return self._process_request(request_type, params)
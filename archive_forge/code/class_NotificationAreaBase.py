from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import param
from bokeh.models import CustomJS
from ..config import config
from ..reactive import ReactiveHTML
from ..util import classproperty
from .datamodel import _DATA_MODELS, construct_data_model
from .resources import CSS_URLS, bundled_files, get_dist_path
from .state import state
class NotificationAreaBase(ReactiveHTML):
    js_events = param.Dict(default={}, doc="\n        A dictionary that configures notifications for specific Bokeh Document\n        events, e.g.:\n\n          {'connection_lost': {'type': 'error', 'message': 'Connection Lost!', 'duration': 5}}\n\n        will trigger a warning on the Bokeh ConnectionLost event.")
    notifications = param.List(item_type=Notification)
    position = param.Selector(default='bottom-right', objects=['bottom-right', 'bottom-left', 'bottom-center', 'top-left', 'top-right', 'top-center', 'center-center', 'center-left', 'center-right'])
    _clear = param.Integer(default=0)
    _notification_type = Notification
    _template = '\n    <div id="pn-notifications" class="notifications" style="position: absolute; bottom: 0; ${position}: 0;">\n    ${notifications}\n    </div>\n    '
    _extension_name = 'notifications'
    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._notification_watchers = {}

    def get_root(self, doc: Optional[Document]=None, comm: Optional[Comm]=None, preprocess: bool=True) -> 'Model':
        root = super().get_root(doc, comm, preprocess)
        for event, notification in self.js_events.items():
            doc.js_on_event(event, CustomJS(code=f'\n            var config = {{\n              message: {notification['message']!r},\n              duration: {notification.get('duration', 0)},\n              notification_type: {notification['type']!r},\n              _destroyed: false\n            }}\n            notifications.data.notifications.push(config)\n            notifications.data.properties.notifications.change.emit()\n            ', args={'notifications': root}))
        self._documents[doc] = root
        state._views[root.ref['id']] = (self, root, doc, comm)
        return root

    def send(self, message, duration=3000, type=None, background=None, icon=None):
        """
        Sends a notification to the frontend.
        """
        notification = self._notification_type(message=message, duration=duration, notification_type=type, notification_area=self, background=background, icon=icon)
        self._notification_watchers[notification] = notification.param.watch(self._remove_notification, '_destroyed')
        self.notifications.append(notification)
        self.param.trigger('notifications')
        return notification

    def error(self, message, duration=3000):
        return self.send(message, duration, type='error')

    def info(self, message, duration=3000):
        return self.send(message, duration, type='info')

    def success(self, message, duration=3000):
        return self.send(message, duration, type='success')

    def warning(self, message, duration=3000):
        return self.send(message, duration, type='warning')

    def clear(self):
        self._clear += 1
        self.notifications[:] = []

    def _remove_notification(self, event):
        if event.obj in self.notifications:
            self.notifications.remove(event.obj)
        event.obj.param.unwatch(self._notification_watchers.pop(event.obj))
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
class NotificationArea(NotificationAreaBase):
    types = param.List(default=[{'type': 'warning', 'background': '#ffc107', 'icon': {'className': 'fas fa-exclamation-triangle', 'tagName': 'i', 'color': 'white'}}, {'type': 'info', 'background': '#007bff', 'icon': {'className': 'fas fa-info-circle', 'tagName': 'i', 'color': 'white'}}])
    __javascript_raw__ = [f'{config.npm_cdn}/notyf@3/notyf.min.js']

    @classproperty
    def __javascript__(cls):
        return bundled_files(cls)

    @classproperty
    def __js_skip__(cls):
        return {'Notyf': cls.__javascript__}
    __js_require__ = {'paths': {'notyf': __javascript_raw__[0][:-3]}}
    __css_raw__ = [f'{config.npm_cdn}/notyf@3/notyf.min.css', CSS_URLS['font-awesome']]

    @classproperty
    def __css__(cls):
        return bundled_files(cls, 'css') + [f'{get_dist_path()}css/notifications.css']
    _template = ''
    _scripts = {'render': "\n        var [y, x] = data.position.split('-')\n        state.toaster = new Notyf({\n          dismissible: true,\n          position: {x: x, y: y},\n          types: data.types\n        })\n      ", 'notifications': "\n        var notification = state.current || data.notifications[data.notifications.length-1]\n        if (notification._destroyed) {\n          return\n        }\n        var config = {\n          duration: notification.duration,\n          type: notification.notification_type,\n          message: notification.message\n        }\n        if (notification.background != null) {\n          config.background = notification.background;\n        }\n        if (notification.icon != null) {\n          config.icon = notification.icon;\n        }\n        var toast = state.toaster.open(config);\n        function destroy() {\n          if (state.current !== notification) {\n            notification._destroyed = true;\n          }\n        }\n        toast.on('dismiss', destroy)\n        if (notification.duration) {\n          setTimeout(destroy, notification.duration)\n        }\n        if (notification.properties === undefined)\n          return\n        view.connect(notification.properties._destroyed.change, function () {\n          state.toaster.dismiss(toast)\n        })\n      ", '_clear': 'state.toaster.dismissAll()', 'position': "\n        script('_clear');\n        script('render');\n        for (notification of data.notifications) {\n          state.current = notification;\n          script('notifications');\n        }\n        state.current = undefined\n      "}

    @classmethod
    def demo(cls):
        """
        Generates a layout which allows demoing the component.
        """
        from ..layout import Column
        from ..widgets import Button, ColorPicker, NumberInput, Select, TextInput
        msg = TextInput(name='Message', value='This is a message')
        duration = NumberInput(name='Duration', value=0, end=10000)
        ntype = Select(name='Type', options=['info', 'warning', 'error', 'success', 'custom'], value='info')
        background = ColorPicker(name='Color', value='#000000')
        button = Button(name='Notify')
        notifications = cls()
        button.js_on_click(args={'notifications': notifications, 'msg': msg, 'duration': duration, 'ntype': ntype, 'color': background}, code="\n            var config = {\n              message: msg.value,\n              duration: duration.value,\n              notification_type: ntype.value,\n              _destroyed: false\n            }\n            if (ntype.value === 'custom') {\n              config.background = color.color\n            }\n            notifications.data.notifications.push(config)\n            notifications.data.properties.notifications.change.emit()\n            ")
        return Column(msg, duration, ntype, background, button, notifications)